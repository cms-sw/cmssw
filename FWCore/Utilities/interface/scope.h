#ifndef FWCOre_Utilities_scope_h
#define FWCOre_Utilities_scope_h

/*****************************************************************************
 *
 * This file is derived from libindi-scope.
 * https://github.com/DarkerStar/libindi-scope
 *
 * It has been modified to conform to CMS conventions
 *
 * libindi-scope is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libindi-scope is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libindi-scope.  If not, see <https://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/*****************************************************************************
 * Scope guards
 *
 * Scope guards are objects that wrap a function, and call that function when
 * the scope guard goes out of scope, depending on certain conditions.
 *
 * The three primary types of scope guards are:
 *  *   scope_exit :    calls the function whenever the guard goes out of
 *                      scope, regardless of why.
 *  *   scope_success : calls the function only if the guard goes out of scope
 *                      normally (not via stack unwinding).
 *  *   scope_fail :    calls the function only if the guard goes out of scope
 *                      via stack unwinding.
 *
 * All scope guards also have a `release()` function, that prevents the
 * wrapped function from being called when the guard goes out of scope.
 *
 * Scope guards cannot be copied, and can only be move constructed (not move
 * assigned). They cannot be default constructed, and can only be constructed
 * with a function object or lambda, or a lvalue reference to a function
 * object or lambda, or a lvalue reference to a function.
 *
 * Usage:
 *      auto f()
 *      {
 *          auto const s1 = scope_exit   {[] { std::cout << "exit!"; }};
 *          auto const s2 = scope_success{[] { std::cout << "good!"; }};
 *          auto const s3 = scope_fail   {[] { std::cout << "fail!"; }};
 *
 *          // [...]
 *
 *          // "fail!" will be printed ONLY if an exception was thrown above.
 *          // "good!" will be printed ONLY if an exception was *NOT* thrown.
 *          // "exit!" will be printed, no matter what happened above.
 *      }
 *
 * Basic interface:
 *      template <typename EF>
 *      class ScopeGuard
 *      {
 *      public:
 *          template <typename EFP>
 *          explicit ScopeGuard(EFP&&) noexcept(*1);
 *
 *          ScopeGuard(ScopeGuard&&) noexcept(*2);
 *
 *          auto release() noexcept -> void;
 *
 *          // No copy construction.
 *          ScopeGuard(ScopeGuard const&) = delete;
 *
 *          // No assignment (no copy assignment OR move assignment).
 *          auto operator=(ScopeGuard const&) -> ScopeGuard& = delete;
 *          auto operator=(ScopeGuard&&)      -> ScopeGuard& = delete;
 *      };
 *
 *      template <typename EF>
 *      ScopeGuard(EF) -> ScopeGuard<EF>;
 *
 * Requirements:
 *      *   (std::is_object_v<EF> and std::is_destructible_v<EF>)
 *              or std::is_lvalue_reference_v<EF>
 *      *   std::is_invocable_v<std::remove_reference_t<EF>>
 *      *   If `g` is an instance of `remove_reference_t<EF>`, `g()` should be
 *          well-formed.
 *
 * Notes:
 *      *1  :   std::is_nothrow_constructible_v<EF, EFP>
 *                  or std::is_nothrow_constructible_v<EF, EFP&>
 *      *2  :   std::is_nothrow_move_constructible_v<EF>
 *                  or std::is_nothrow_copy_constructible_v<EF>
 *
 * Specific scope guards may have additional or slightly modified
 * requirements.
 *
 * This header is based on the proposed extension to the C++ standard library
 * P0052 and std::experimental::scope_exit.
 *
 * This header is currently based on revision 10 of P0052, found at:
 *     http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0052r10.pdf
 *
 ****************************************************************************/

#include <limits>
#include <type_traits>
#include <utility>

namespace edm {
  inline namespace v1 {

    // From P0550
    template <class T>
    using remove_cvref = std::remove_cv<std::remove_reference<T>>;

    template <class T>
    using remove_cvref_t = typename remove_cvref<T>::type;

    namespace _detail_X_scope {

      // move_init_if_noexcept<T, U>(U&&)
      //
      // Works almost identically to `std::forward<U>(u)`, except that if
      // constructing a `T` from a `U&&` may throw an exception, it will return a
      // lvalue reference (not a rvalue reference).
      //
      // When used as the initializer of a data-member:
      //      template <typename U>
      //      type(U&& u) : _t{move_init_if_noexcept<T, U>(u)} {}
      // it will construct `_t` with an rvalue if and only if `U` is not an lvalue
      // reference AND the construction will not throw. Otherwise, it will construct
      // `_t` with an lvalue. (In other words, it will only move-construct `_t` if
      // that will not throw, otherwise it will copy-construct `_t`.)
      template <typename T, typename U>
      constexpr auto move_init_if_noexcept(U& u) noexcept -> decltype(auto) {
        // If U is a lvalue reference, we can't move-construct in any case, so
        // return a lvalue reference.
        if constexpr (std::is_lvalue_reference_v<U>) {
          return static_cast<std::remove_reference_t<U>&>(u);
        } else {
          // U is an rvalue, so we can potentially move-construct a T (by
          // returning a rvalue reference).
          //
          // However, if that operation would *NOT* be noexcept, do a
          // copy-construct (by returning an lvalue reference) instead.
          if constexpr (std::is_nothrow_constructible_v<T, U>)
            return static_cast<std::remove_reference_t<U>&&>(u);
          else
            return static_cast<std::remove_reference_t<U>&>(u);
        }
      }

      // scope_guard_base<EF>
      //
      // Base type for scope guards, to set up some sensible defaults and avoid
      // repetition.
      template <typename EF>
      class scope_guard_base {
      public:
        // 7.5.2.3 requirements.
        static_assert((std::is_object_v<EF> and std::is_destructible_v<EF>) or std::is_lvalue_reference_v<EF>);
        static_assert(std::is_invocable_v<std::remove_reference_t<EF>>);

        // Scope guards are move constructible.
        constexpr scope_guard_base(scope_guard_base&&) noexcept = default;

        // Scope guards are destructible.
        ~scope_guard_base() = default;

        // Scope guards are non-copyable.
        scope_guard_base(scope_guard_base const&) = delete;
        auto operator=(scope_guard_base const&) -> scope_guard_base& = delete;

        // Scope guards have no move-assignment.
        auto operator=(scope_guard_base&&) -> scope_guard_base& = delete;

      protected:
        // Only derived types (which should be scope guards) can construct.
        constexpr scope_guard_base() noexcept = default;
      };

    }  // namespace _detail_X_scope

    // scope_exit<EF>
    //
    // scope_exit is a scope guard that calls its contained function whenever the
    // scope exits, whether that exit is a successful (normal) exit or a failure
    // (via stack unwinding) exit.
    //
    // Extra requirements (in addition to basic scope guard requirements):
    //      *   If `g` is an instance of `remove_reference_t<EF>`, `g()` should
    //          not raise an exception.
    template <typename EF>
    class scope_exit : public _detail_X_scope::scope_guard_base<EF> {
    public:
      template <typename EFP>
      explicit scope_exit(EFP&& f) noexcept(std::is_nothrow_constructible_v<EF, EFP> or
                                            std::is_nothrow_constructible_v<EF, EFP&>) try
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EFP>(f)}, _execute_on_destruction{true} {
        // 7.5.2.5 requirements.
        static_assert(not std::is_same_v<remove_cvref_t<EFP>, scope_exit>);
        static_assert(std::is_nothrow_constructible_v<EF, EFP> or std::is_constructible_v<EF, EFP&>);
      } catch (...) {
        f();
      }

      scope_exit(scope_exit&& other) noexcept(std::is_nothrow_move_constructible_v<EF> or
                                              std::is_nothrow_copy_constructible_v<EF>)
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EF&&>(other._exit_function)},
            _execute_on_destruction{other._execute_on_destruction} {
        other.release();
      }

      ~scope_exit() {
        if (_execute_on_destruction)
          _exit_function();
      }

      auto release() noexcept -> void { _execute_on_destruction = false; }

    private:
      EF _exit_function;
      bool _execute_on_destruction = true;
    };

    template <typename EF>
    scope_exit(EF) -> scope_exit<EF>;

    // scope_fail<EF>
    //
    // scope_fail is a scope guard that calls its contained function only in
    // the case that it is destroyed during stack unwinding.
    template <typename EF>
    class scope_fail : public _detail_X_scope::scope_guard_base<EF> {
    public:
      template <typename EFP>
      explicit scope_fail(EFP&& f) noexcept(std::is_nothrow_constructible_v<EF, EFP> or
                                            std::is_nothrow_constructible_v<EF, EFP&>) try
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EFP>(f)},
            _uncaught_on_creation{std::uncaught_exceptions()} {
        // 7.5.2.10 requirements.
        static_assert(not std::is_same_v<remove_cvref_t<EFP>, scope_fail>);
        static_assert(std::is_nothrow_constructible_v<EF, EFP> or std::is_constructible_v<EF, EFP&>);
      } catch (...) {
        f();
      }

      scope_fail(scope_fail&& other) noexcept(std::is_nothrow_move_constructible_v<EF> or
                                              std::is_nothrow_copy_constructible_v<EF>)
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EF&&>(other._exit_function)},
            _uncaught_on_creation{other._uncaught_on_creation} {
        other.release();
      }

      ~scope_fail() {
        if (std::uncaught_exceptions() > _uncaught_on_creation)
          _exit_function();
      }

      auto release() noexcept -> void {
        // The number of uncaught exceptions can never be greater than the
        // max value of int, so by setting the count to this, the destructor
        // condition can never be met.
        _uncaught_on_creation = std::numeric_limits<int>::max();
      }

    private:
      EF _exit_function;
      int _uncaught_on_creation = 0;
    };

    template <typename EF>
    scope_fail(EF) -> scope_fail<EF>;

    // scope_success<EF>
    //
    // scope_success is a scope guard that calls its contained function only in
    // the case that it is destroyed under normal conditions (not stack
    // unwinding).
    template <typename EF>
    class scope_success : public _detail_X_scope::scope_guard_base<EF> {
    public:
      template <typename EFP>
      explicit scope_success(EFP&& f) noexcept(std::is_nothrow_constructible_v<EF, EFP> or
                                               std::is_nothrow_constructible_v<EF, EFP&>)
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EFP>(f)},
            _uncaught_on_creation{std::uncaught_exceptions()} {
        // 7.5.2.15 requirements.
        static_assert(not std::is_same_v<remove_cvref_t<EFP>, scope_success>);
        static_assert(std::is_nothrow_constructible_v<EF, EFP> or std::is_constructible_v<EF, EFP&>);
      }

      scope_success(scope_success&& other) noexcept(std::is_nothrow_move_constructible_v<EF> or
                                                    std::is_nothrow_copy_constructible_v<EF>)
          : _exit_function{_detail_X_scope::move_init_if_noexcept<EF, EF&&>(other._exit_function)},
            _uncaught_on_creation{other._uncaught_on_creation} {
        other.release();
      }

      ~scope_success() noexcept(noexcept(_exit_function())) {
        if (std::uncaught_exceptions() <= _uncaught_on_creation)
          _exit_function();
      }

      auto release() noexcept -> void {
        // The number of uncaught exceptions can never be less than zero,
        // so by setting the count to -1, the destructor condition can never
        // be met.
        _uncaught_on_creation = -1;
      }

    private:
      EF _exit_function;
      int _uncaught_on_creation = 0;
    };

    template <typename EF>
    scope_success(EF) -> scope_success<EF>;

  }  // namespace v1
}  // namespace edm

#endif  // include guard

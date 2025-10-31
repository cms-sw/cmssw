#ifndef DataFormats_SoATemplate_interface_SoABlocks_h
#define DataFormats_SoATemplate_interface_SoABlocks_h

/*
 * SoA Blocks: collection of SoA layouts (blocks) that can be accessed in a structured way.
 */

#include "SoALayout.h"
#include "SoACommon.h"

/*
 * Declare accessors for the View of each block
 */
#define _DECLARE_ACCESSORS_VIEW_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME)                \
  SOA_HOST_DEVICE SOA_INLINE LAYOUT_NAME<ALIGNMENT>::View NAME() {                        \
    return LAYOUT_NAME<ALIGNMENT>::const_cast_View(base_type::BOOST_PP_CAT(NAME, View_)); \
  }

#define _DECLARE_ACCESSORS_VIEW_BLOCKS(R, DATA, NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_ACCESSORS_VIEW_BLOCKS_IMPL NAME))

/*
 * Declare parameters for contructing the View of an SoA by blocks
 * using different views for each block
 */
#define _DECLARE_VIEW_CONSTRUCTOR_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) (LAYOUT_NAME<ALIGNMENT>::View NAME)

#define _DECLARE_VIEW_CONSTRUCTOR_BLOCKS(R, DATA, NAME)                          \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_VIEW_CONSTRUCTOR_BLOCKS_IMPL NAME))

/*
 * Build the View of each block
 */
#define _INITIALIZE_MEMBER_VIEW_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) (NAME)

#define _INITIALIZE_MEMBER_VIEW_BLOCKS(R, DATA, NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_INITIALIZE_MEMBER_VIEW_BLOCKS_IMPL NAME))

/**
 * Pointers to blocks for referencing by name
 */
#define _DECLARE_BLOCKS_POINTERS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME)              \
  SOA_HOST_DEVICE SOA_INLINE auto const* BOOST_PP_CAT(addressOf_, NAME)() const { \
    return parent_.BOOST_PP_CAT(NAME, _).metadata().data();                       \
  }                                                                               \
  SOA_HOST_DEVICE SOA_INLINE auto* BOOST_PP_CAT(addressOf_, NAME)() {             \
    return parent_.BOOST_PP_CAT(NAME, _).metadata().data();                       \
  }

#define _DECLARE_BLOCKS_POINTERS(R, DATA, TYPE_NAME)                                  \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                       \
              BOOST_PP_EXPAND(_DECLARE_BLOCKS_POINTERS_IMPL TYPE_NAME))

/*
 * Declare accessors for the ConstView of each block
 */
#define _DECLARE_ACCESSORS_CONST_VIEW_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  SOA_HOST_DEVICE SOA_INLINE const LAYOUT_NAME<ALIGNMENT>::ConstView NAME() const { return BOOST_PP_CAT(NAME, View_); }

#define _DECLARE_ACCESSORS_CONST_VIEW_BLOCKS(R, DATA, NAME)                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_ACCESSORS_CONST_VIEW_BLOCKS_IMPL NAME))

/*
 * Build the ConstView of each block from the corresponding Layout
 */
#define _DECLARE_MEMBER_CONST_VIEW_CONSTRUCTION_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  (BOOST_PP_CAT(NAME, View_)(blocks.BOOST_PP_CAT(NAME, _)))

#define _DECLARE_MEMBER_CONST_VIEW_CONSTRUCTION_BLOCKS(R, DATA, NAME)            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_CONST_VIEW_CONSTRUCTION_BLOCKS_IMPL NAME))

/*
 * Declare parameters for contructing the ConstView of an SoA by blocks
 * using different const views for each block
 */
#define _DECLARE_CONST_VIEW_CONSTRUCTOR_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  (LAYOUT_NAME<ALIGNMENT>::ConstView NAME)

#define _DECLARE_CONST_VIEW_CONSTRUCTOR_BLOCKS(R, DATA, NAME)                    \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_CONSTRUCTOR_BLOCKS_IMPL NAME))

/*
 * Build the ConstView of each block
 */
#define _INITIALIZE_MEMBER_CONST_VIEW_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) (BOOST_PP_CAT(NAME, View_){NAME})

#define _INITIALIZE_MEMBER_CONST_VIEW_BLOCKS(R, DATA, NAME)                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_INITIALIZE_MEMBER_CONST_VIEW_BLOCKS_IMPL NAME))

/*
 * Initialize the array of sizes for the View of an SoA by blocks
 */
#define _DECLARE_CONST_VIEW_SIZES_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) (BOOST_PP_CAT(NAME, View_).metadata().size())

#define _DECLARE_CONST_VIEW_SIZES(R, DATA, NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_SIZES_IMPL NAME))

/*
 * Declare the data members for the ConstView of the SoA by blocks
 */
#define _DECLARE_MEMBERS_CONST_VIEW_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  LAYOUT_NAME<ALIGNMENT>::ConstView BOOST_PP_CAT(NAME, View_);

#define _DECLARE_MEMBERS_CONST_VIEW_BLOCKS(R, DATA, NAME)                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_MEMBERS_CONST_VIEW_BLOCKS_IMPL NAME))

/*
 * Declare accessors for the Layout of each block
 */
#define _DECLARE_LAYOUTS_ACCESSORS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  LAYOUT_NAME<ALIGNMENT> NAME() { return BOOST_PP_CAT(NAME, _); }

#define _DECLARE_LAYOUTS_ACCESSORS(R, DATA, NAME)                                \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_LAYOUTS_ACCESSORS_IMPL NAME))

/*
 * Computation of the size for each block
 */
#define _ACCUMULATE_SOA_BLOCKS_SIZE_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME)   \
  _soa_impl_ret += LAYOUT_NAME<ALIGNMENT>::computeDataSize(sizes[index]); \
  index++;

#define _ACCUMULATE_SOA_BLOCKS_SIZE(R, DATA, NAME)                               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_ACCUMULATE_SOA_BLOCKS_SIZE_IMPL NAME))

/*
 * Computation of the block location in the memory layout (at SoA by blocks construction time)
 */
#define _DECLARE_MEMBER_CONSTRUCTION_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  BOOST_PP_CAT(NAME, _) = LAYOUT_NAME<ALIGNMENT>(mem + offset, sizes_[index]);  \
  offset += LAYOUT_NAME<ALIGNMENT>::computeDataSize(sizes_[index]);             \
  index++;

#define _DECLARE_MEMBER_CONSTRUCTION_BLOCKS(R, DATA, NAME)                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_CONSTRUCTION_BLOCKS_IMPL NAME))

/*
 * Call default constructor for each block
 */
#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) (BOOST_PP_CAT(NAME, _)())

#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_BLOCKS(R, DATA, NAME)               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_BLOCKS_IMPL NAME))

/*
 * Computate number of blocks
 */
#define _COUNT_SOA_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) soa_blocks_count += 1;

#define _COUNT_SOA_BLOCKS(R, DATA, NAME)                                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_COUNT_SOA_BLOCKS_IMPL NAME))

/*
 * Call the copy constructor for each block
 */
#define _DECLARE_BLOCK_MEMBER_COPY_CONSTRUCTION_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  (BOOST_PP_CAT(NAME, _)(_soa_impl_other.BOOST_PP_CAT(NAME, _)))

#define _DECLARE_BLOCK_MEMBER_COPY_CONSTRUCTION(R, DATA, NAME)                   \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_BLOCK_MEMBER_COPY_CONSTRUCTION_IMPL NAME))

/*
 * Call the assignment operator for each block
 */
#define _DECLARE_BLOCKS_MEMBER_ASSIGNMENT_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);

#define _DECLARE_BLOCKS_MEMBER_ASSIGNMENT(R, DATA, NAME)                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_BLOCKS_MEMBER_ASSIGNMENT_IMPL NAME))

/*
 * Check the equality of sizes for each block
 */
#define _CHECK_VIEW_SIZES_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME, CLASS_NAME)                                             \
  if (sizes_[index] < view.NAME().metadata().size()) {                                                                \
    throw std::runtime_error("In " #CLASS_NAME "::deepCopy method: number of elements mismatch for block " #NAME ""); \
  }                                                                                                                   \
  index++;

#define _CHECK_VIEW_SIZES(R, DATA, NAME)                                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_CHECK_VIEW_SIZES_IMPL BOOST_PP_TUPLE_PUSH_BACK(NAME, DATA)))

/*
 * Call the deepCopy method for each block
 */
#define _DEEP_COPY_VIEW_COLUMNS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) BOOST_PP_CAT(NAME, _).deepCopy(view.NAME());

#define _DEEP_COPY_VIEW_COLUMNS(R, DATA, NAME)                                   \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DEEP_COPY_VIEW_COLUMNS_IMPL NAME))

/*
 * Call ROOTReadstreamer for each block.
 */
#define _STREAMER_READ_SOA_BLOCK_DATA_MEMBER_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  BOOST_PP_CAT(NAME, _).ROOTReadStreamer(onfile);

#define _STREAMER_READ_SOA_BLOCK_DATA_MEMBER(R, DATA, NAME)                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_STREAMER_READ_SOA_BLOCK_DATA_MEMBER_IMPL NAME))

/*
 * Call ROOTStreamerCleaner for each block.
 */
#define _ROOT_FREE_SOA_BLOCK_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) \
  BOOST_PP_CAT(NAME, _).ROOTStreamerCleaner();

#define _ROOT_FREE_SOA_BLOCK_COLUMN_OR_SCALAR(R, DATA, NAME)                     \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_ROOT_FREE_SOA_BLOCK_COLUMN_OR_SCALAR_IMPL NAME))

#define _DECLARE_MEMBERS_BLOCKS_IMPL(VALUE_TYPE, NAME, LAYOUT_NAME) LAYOUT_NAME<ALIGNMENT> BOOST_PP_CAT(NAME, _);

/*
 * Declare the data members for the SoA by blocks
 */
#define _DECLARE_MEMBERS_BLOCKS(R, DATA, NAME)                                   \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, NAME), _VALUE_TYPE_BLOCK), \
              BOOST_PP_EMPTY(),                                                  \
              BOOST_PP_EXPAND(_DECLARE_MEMBERS_BLOCKS_IMPL NAME))

/*
 * A macro defining a SoA by blocks layout (collection of SoA layouts)
 */
// clang-format off
#define GENERATE_SOA_BLOCKS(CLASS, ...)                                                                                \
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                                   \
            bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed>                                      \
  struct CLASS {                                                                                                       \
    using size_type = cms::soa::size_type;                                                                             \
    using byte_size_type = cms::soa::byte_size_type;                                                                   \
    constexpr static byte_size_type alignment = ALIGNMENT;                                                             \
                                                                                                                       \
    struct ConstView;                                                                                                  \
    struct View;                                                                                                       \
                                                                                                                       \
    /* Helper function to compute the total number of blocks */                                                        \
    static constexpr size_type computeBlocksNumber() {                                                                 \
      size_type soa_blocks_count = 0;                                                                                  \
      _ITERATE_ON_ALL(_COUNT_SOA_BLOCKS, ~, __VA_ARGS__)                                                               \
      return soa_blocks_count;                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    static constexpr size_type blocksNumber = computeBlocksNumber();                                                   \
                                                                                                                       \
    /* Helper function used by caller to externally allocate the storage */                                            \
    static constexpr byte_size_type computeDataSize(std::array<size_type, blocksNumber> sizes) {                       \
      byte_size_type _soa_impl_ret = 0;                                                                                \
      size_type index = 0;                                                                                             \
      _ITERATE_ON_ALL(_ACCUMULATE_SOA_BLOCKS_SIZE, ~, __VA_ARGS__)                                                     \
      return _soa_impl_ret;                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /**                                                                                                                \
     * Helper/friend class allowing SoA by blocks introspection.                                                       \
     */                                                                                                                \
    struct Metadata {                                                                                                  \
      friend CLASS;                                                                                                    \
      SOA_HOST_DEVICE SOA_INLINE std::array<size_type, blocksNumber> size() const { return parent_.sizes_; }           \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type byteSize() const { return CLASS::computeDataSize(parent_.sizes_); }    \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type alignment() const { return CLASS::alignment; }                         \
      SOA_HOST_DEVICE SOA_INLINE CLASS cloneToNewAddress(std::byte* _soa_impl_addr) const {                            \
        return CLASS(_soa_impl_addr, parent_.sizes_);                                                                  \
      }                                                                                                                \
                                                                                                                       \
      /* Pointers to each block */                                                                                     \
      _ITERATE_ON_ALL(_DECLARE_BLOCKS_POINTERS, ~, __VA_ARGS__)                                                        \
                                                                                                                       \
      Metadata& operator=(const Metadata&) = delete;                                                                   \
      Metadata(const Metadata&) = delete;                                                                              \
                                                                                                                       \
      private:                                                                                                         \
        SOA_HOST_DEVICE SOA_INLINE Metadata(const CLASS& _soa_impl_parent) : parent_(_soa_impl_parent) {}              \
        const CLASS& parent_;                                                                                          \
    };                                                                                                                 \
                                                                                                                       \
    friend Metadata;                                                                                                   \
                                                                                                                       \
    SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                             \
    SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                         \
                                                                                                                       \
    _ITERATE_ON_ALL(_DECLARE_LAYOUTS_ACCESSORS, ~, __VA_ARGS__)                                                        \
                                                                                                                       \
    struct ConstView {                                                                                                 \
      friend struct View;                                                                                              \
      /* Helper/friend class allowing SoA by blocks ConstView introspection. */                                        \
      struct Metadata {                                                                                                \
        friend ConstView;                                                                                              \
        SOA_HOST_DEVICE SOA_INLINE std::array<size_type, blocksNumber> size() const { return parent_.sizes_; }         \
                                                                                                                       \
        /* Forbid copying to avoid const correctness evasion */                                                        \
        Metadata& operator=(const Metadata&) = delete;                                                                 \
        Metadata(const Metadata&) = delete;                                                                            \
                                                                                                                       \
      private:                                                                                                         \
        SOA_HOST_DEVICE SOA_INLINE Metadata(const ConstView& _soa_impl_parent)                                         \
          : parent_(_soa_impl_parent) {}                                                                               \
        const ConstView& parent_;                                                                                      \
      };                                                                                                               \
                                                                                                                       \
      friend Metadata;                                                                                                 \
      SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                           \
                                                                                                                       \
      /* Trivial constuctor */                                                                                         \
      ConstView() = default;                                                                                           \
                                                                                                                       \
      /* Copiable */                                                                                                   \
      ConstView(ConstView const&) = default;                                                                           \
      ConstView& operator=(ConstView const&) = default;                                                                \
                                                                                                                       \
      /* Movable */                                                                                                    \
      ConstView(ConstView &&) = default;                                                                               \
      ConstView& operator=(ConstView &&) = default;                                                                    \
                                                                                                                       \
      /* Trivial destuctor */                                                                                          \
      ~ConstView() = default;                                                                                          \
                                                                                                                       \
      /* Constructor relying on user provided Layout by blocks */                                                      \
      SOA_HOST_ONLY ConstView(CLASS& blocks)                                                                           \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_CONST_VIEW_CONSTRUCTION_BLOCKS, ~, __VA_ARGS__),                     \
            sizes_{blocks.sizes_} {}                                                                                   \
                                                                                                                       \
      /* Constructor relying on user provided const views for each block */                                            \
      SOA_HOST_ONLY ConstView(_ITERATE_ON_ALL_COMMA(_DECLARE_CONST_VIEW_CONSTRUCTOR_BLOCKS, ~, __VA_ARGS__))           \
          : _ITERATE_ON_ALL_COMMA(_INITIALIZE_MEMBER_CONST_VIEW_BLOCKS, ~, __VA_ARGS__),                               \
            sizes_{{_ITERATE_ON_ALL_COMMA(_DECLARE_CONST_VIEW_SIZES, ~, __VA_ARGS__)}} {}                              \
                                                                                                                       \
      /* Accessors for the const views for each block */                                                               \
      _ITERATE_ON_ALL(_DECLARE_ACCESSORS_CONST_VIEW_BLOCKS, ~, __VA_ARGS__)                                            \
                                                                                                                       \
      private:                                                                                                         \
        _ITERATE_ON_ALL(_DECLARE_MEMBERS_CONST_VIEW_BLOCKS, ~, __VA_ARGS__)                                            \
        std::array<size_type, blocksNumber> sizes_;                                                                    \
    };                                                                                                                 \
                                                                                                                       \
    struct View : ConstView {                                                                                          \
      using base_type = ConstView;                                                                                     \
      /* Helper/friend class allowing SoA by blocks View introspection. */                                             \
      struct Metadata {                                                                                                \
        friend View;                                                                                                   \
        SOA_HOST_DEVICE SOA_INLINE std::array<size_type, blocksNumber> size() const { return parent_.sizes_; }         \
                                                                                                                       \
        /* Forbid copying to avoid const correctness evasion */                                                        \
        Metadata& operator=(const Metadata&) = delete;                                                                 \
        Metadata(const Metadata&) = delete;                                                                            \
                                                                                                                       \
      private:                                                                                                         \
        SOA_HOST_DEVICE SOA_INLINE Metadata(const View& _soa_impl_parent)                                              \
          : parent_(_soa_impl_parent) {}                                                                               \
        const View& parent_;                                                                                           \
      };                                                                                                               \
                                                                                                                       \
      friend Metadata;                                                                                                 \
      SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                       \
                                                                                                                       \
      /* Trivial constuctor */                                                                                         \
      View() = default;                                                                                                \
                                                                                                                       \
      /* Copiable */                                                                                                   \
      View(View const&) = default;                                                                                     \
      View& operator=(View const&) = default;                                                                          \
                                                                                                                       \
      /* Movable */                                                                                                    \
      View(View &&) = default;                                                                                         \
      View& operator=(View &&) = default;                                                                              \
                                                                                                                       \
      /* Trivial destuctor */                                                                                          \
      ~View() = default;                                                                                               \
                                                                                                                       \
      /* Constructor relying on user provided Layout by blocks */                                                      \
      SOA_HOST_ONLY View(CLASS& blocks)                                                                                \
          : base_type{blocks} {}                                                                                       \
                                                                                                                       \
      /* Constructor relying on user provided views for each block */                                                  \
      SOA_HOST_ONLY View(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTOR_BLOCKS, ~, __VA_ARGS__)) :                    \
        base_type{_ITERATE_ON_ALL_COMMA(_INITIALIZE_MEMBER_VIEW_BLOCKS, ~, __VA_ARGS__)} {}                            \
                                                                                                                       \
      /* Accessors for the views for each block */                                                                     \
      _ITERATE_ON_ALL(_DECLARE_ACCESSORS_VIEW_BLOCKS, ~, __VA_ARGS__)                                                  \
                                                                                                                       \
       /* Data members inherited from the ConstView */                                                                 \
    };                                                                                                                 \
                                                                                                                       \
    /* TODO: implement Descriptor and ConstDescriptor for Blocks to enable heterogeneous deepCopy */                   \
    struct Descriptor;                                                                                                 \
    struct ConstDescriptor;                                                                                            \
                                                                                                                       \
    /* Trivial constuctor */                                                                                           \
    CLASS()                                                                                                            \
        : sizes_{},                                                                                                    \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_BLOCKS, ~, __VA_ARGS__) {}                        \
                                                                                                                       \
    /* Constructor relying on user provided storage and array of sizes */                                              \
    SOA_HOST_ONLY CLASS(std::byte* mem, std::array<size_type, blocksNumber> elements) : sizes_(elements) {             \
      byte_size_type offset = 0;                                                                                       \
      size_type index = 0;                                                                                             \
      _ITERATE_ON_ALL(_DECLARE_MEMBER_CONSTRUCTION_BLOCKS, ~, __VA_ARGS__)                                             \
    }                                                                                                                  \
                                                                                                                       \
    /* Explicit copy constructor and assignment operator */                                                            \
    SOA_HOST_ONLY CLASS(CLASS const& _soa_impl_other)                                                                  \
        : sizes_(_soa_impl_other.sizes_),                                                                              \
          _ITERATE_ON_ALL_COMMA(_DECLARE_BLOCK_MEMBER_COPY_CONSTRUCTION, ~, __VA_ARGS__) {}                            \
                                                                                                                       \
    SOA_HOST_ONLY CLASS& operator=(CLASS const& _soa_impl_other) {                                                     \
      sizes_ = _soa_impl_other.sizes_;                                                                                 \
      _ITERATE_ON_ALL(_DECLARE_BLOCKS_MEMBER_ASSIGNMENT, ~, __VA_ARGS__)                                               \
      return *this;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * Method for copying the data from a generic ConstView by blocks to a memory blob.                                \
     * Host-only data can be handled by this method.                                                                   \
     */                                                                                                                \
    SOA_HOST_ONLY void deepCopy(ConstView const& view) {                                                               \
      size_type index = 0;                                                                                             \
      _ITERATE_ON_ALL(_CHECK_VIEW_SIZES, CLASS, __VA_ARGS__)                                                           \
      _ITERATE_ON_ALL(_DEEP_COPY_VIEW_COLUMNS, ~, __VA_ARGS__)                                                         \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT read streamer */                                                                                           \
    template <typename T>                                                                                              \
    void ROOTReadStreamer(T & onfile) {                                                                                \
      _ITERATE_ON_ALL(_STREAMER_READ_SOA_BLOCK_DATA_MEMBER, ~, __VA_ARGS__)                                            \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT allocation cleanup */                                                                                      \
    void ROOTStreamerCleaner() {                                                                                       \
      /* This function should only be called from the PortableCollection ROOT streamer */                              \
      _ITERATE_ON_ALL(_ROOT_FREE_SOA_BLOCK_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                           \
    }                                                                                                                  \
                                                                                                                       \
    private:                                                                                                           \
      /* Data members */                                                                                               \
      std::array<size_type, blocksNumber> sizes_;                                                                      \
      _ITERATE_ON_ALL(_DECLARE_MEMBERS_BLOCKS, ~, __VA_ARGS__)                                                         \
  }; \
  // clang-format on

#endif  // DataFormats_SoATemplate_interface_SoABlocks_h

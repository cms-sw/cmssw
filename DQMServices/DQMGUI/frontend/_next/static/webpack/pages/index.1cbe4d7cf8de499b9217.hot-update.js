webpackHotUpdate_N_E("pages/index",{

/***/ "./hooks/useRequest.tsx":
/*!******************************!*\
  !*** ./hooks/useRequest.tsx ***!
  \******************************/
/*! exports provided: useRequest */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useRequest", function() { return useRequest; });
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/defineProperty */ "./node_modules/@babel/runtime/helpers/esm/defineProperty.js");
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! axios */ "./node_modules/axios/index.js");
/* harmony import */ var axios__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(axios__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../config/config */ "./config/config.ts");
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../components/utils */ "./components/utils.ts");




var _s = $RefreshSig$();

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_1__["default"])(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }





//for traching, which req. should be canceled
var useRequest = function useRequest(url) {
  _s();

  var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  var watchers = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
  var should_we_fetch = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : true;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(null),
      data = _useState[0],
      setData = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(false),
      isLoading = _useState2[0],
      setIsLoading = _useState2[1];

  var cancelSource = Object(react__WEBPACK_IMPORTED_MODULE_3__["useRef"])(null);

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])([]),
      errors = _useState3[0],
      setErrors = _useState3[1];

  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    if (cancelSource) {
      var _cancelSource$current;

      (_cancelSource$current = cancelSource.current) === null || _cancelSource$current === void 0 ? void 0 : _cancelSource$current.cancel();
    }
  }, []);
  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    var CancelToken = axios__WEBPACK_IMPORTED_MODULE_4___default.a.CancelToken;
    cancelSource.current = CancelToken.source();

    var fetchData = /*#__PURE__*/function () {
      var _ref = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
        var _cancelSource$current2, _cancelSource$current3, basePath, response, _data, _cancelSource$current4;

        return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return setIsLoading(true);

              case 2:
                _context.prev = 2;
                setTimeout((_cancelSource$current2 = cancelSource.current) === null || _cancelSource$current2 === void 0 ? void 0 : _cancelSource$current2.cancel, 180000);
                basePath = Object(_components_utils__WEBPACK_IMPORTED_MODULE_6__["getPathName"])();
                _context.next = 7;
                return axios__WEBPACK_IMPORTED_MODULE_4___default.a.request(_objectSpread({
                  url: "".concat(_config_config__WEBPACK_IMPORTED_MODULE_5__["root_url"]).concat(url),
                  method: options.method || 'get',
                  cancelToken: (_cancelSource$current3 = cancelSource.current) === null || _cancelSource$current3 === void 0 ? void 0 : _cancelSource$current3.token
                }, options));

              case 7:
                response = _context.sent;
                _data = response.data;
                setData(_data);
                setIsLoading(false);
                _context.next = 19;
                break;

              case 13:
                _context.prev = 13;
                _context.t0 = _context["catch"](2);
                setIsLoading(false);
                setErrors([_context.t0.toString()]);

                if (axios__WEBPACK_IMPORTED_MODULE_4___default.a.isCancel(_context.t0)) {
                  setIsLoading(false);
                  setErrors(['Request Timeout']);
                }

                (_cancelSource$current4 = cancelSource.current) === null || _cancelSource$current4 === void 0 ? void 0 : _cancelSource$current4.cancel();

              case 19:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, null, [[2, 13]]);
      }));

      return function fetchData() {
        return _ref.apply(this, arguments);
      };
    }();

    if (should_we_fetch) {
      fetchData();
    }

    return function () {
      return setErrors([]);
    };
  }, watchers);
  return {
    data: data,
    isLoading: isLoading,
    errors: errors,
    cancelSource: cancelSource
  };
};

_s(useRequest, "P5+J8ZtAZmlGTIBAZ9YHYAVhoRQ=");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlUmVxdWVzdC50c3giXSwibmFtZXMiOlsidXNlUmVxdWVzdCIsInVybCIsIm9wdGlvbnMiLCJ3YXRjaGVycyIsInNob3VsZF93ZV9mZXRjaCIsInVzZVN0YXRlIiwiZGF0YSIsInNldERhdGEiLCJpc0xvYWRpbmciLCJzZXRJc0xvYWRpbmciLCJjYW5jZWxTb3VyY2UiLCJ1c2VSZWYiLCJlcnJvcnMiLCJzZXRFcnJvcnMiLCJ1c2VFZmZlY3QiLCJjdXJyZW50IiwiY2FuY2VsIiwiQ2FuY2VsVG9rZW4iLCJheGlvcyIsInNvdXJjZSIsImZldGNoRGF0YSIsInNldFRpbWVvdXQiLCJiYXNlUGF0aCIsImdldFBhdGhOYW1lIiwicmVxdWVzdCIsInJvb3RfdXJsIiwibWV0aG9kIiwiY2FuY2VsVG9rZW4iLCJ0b2tlbiIsInJlc3BvbnNlIiwidG9TdHJpbmciLCJpc0NhbmNlbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBTUE7QUFDQTtBQVNBO0FBQ08sSUFBTUEsVUFBVSxHQUFHLFNBQWJBLFVBQWEsQ0FDeEJDLEdBRHdCLEVBS047QUFBQTs7QUFBQSxNQUhsQkMsT0FHa0IsdUVBSFksRUFHWjtBQUFBLE1BRmxCQyxRQUVrQix1RUFGQSxFQUVBO0FBQUEsTUFEbEJDLGVBQ2tCLHVFQURTLElBQ1Q7O0FBQUEsa0JBQ01DLHNEQUFRLENBQUMsSUFBRCxDQURkO0FBQUEsTUFDWEMsSUFEVztBQUFBLE1BQ0xDLE9BREs7O0FBQUEsbUJBRWdCRixzREFBUSxDQUFDLEtBQUQsQ0FGeEI7QUFBQSxNQUVYRyxTQUZXO0FBQUEsTUFFQUMsWUFGQTs7QUFHbEIsTUFBTUMsWUFBWSxHQUFHQyxvREFBTSxDQUEyQixJQUEzQixDQUEzQjs7QUFIa0IsbUJBSVVOLHNEQUFRLENBQVcsRUFBWCxDQUpsQjtBQUFBLE1BSVhPLE1BSlc7QUFBQSxNQUlIQyxTQUpHOztBQU1sQkMseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBSUosWUFBSixFQUFrQjtBQUFBOztBQUNoQiwrQkFBQUEsWUFBWSxDQUFDSyxPQUFiLGdGQUFzQkMsTUFBdEI7QUFDRDtBQUNGLEdBSlEsRUFJTixFQUpNLENBQVQ7QUFNQUYseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBTUcsV0FBVyxHQUFHQyw0Q0FBSyxDQUFDRCxXQUExQjtBQUNBUCxnQkFBWSxDQUFDSyxPQUFiLEdBQXVCRSxXQUFXLENBQUNFLE1BQVosRUFBdkI7O0FBRUEsUUFBTUMsU0FBUztBQUFBLGtNQUFHO0FBQUE7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLHVCQUNWWCxZQUFZLENBQUMsSUFBRCxDQURGOztBQUFBO0FBQUE7QUFHZFksMEJBQVUsMkJBQUNYLFlBQVksQ0FBQ0ssT0FBZCwyREFBQyx1QkFBc0JDLE1BQXZCLEVBQStCLE1BQS9CLENBQVY7QUFDTU0sd0JBSlEsR0FJR0MscUVBQVcsRUFKZDtBQUFBO0FBQUEsdUJBS3dCTCw0Q0FBSyxDQUFDTSxPQUFOO0FBQ3BDdkIscUJBQUcsWUFBS3dCLHVEQUFMLFNBQWdCeEIsR0FBaEIsQ0FEaUM7QUFFcEN5Qix3QkFBTSxFQUFFeEIsT0FBTyxDQUFDd0IsTUFBUixJQUFrQixLQUZVO0FBR3BDQyw2QkFBVyw0QkFBRWpCLFlBQVksQ0FBQ0ssT0FBZiwyREFBRSx1QkFBc0JhO0FBSEMsbUJBSWpDMUIsT0FKaUMsRUFMeEI7O0FBQUE7QUFLUjJCLHdCQUxRO0FBV052QixxQkFYTSxHQVdHdUIsUUFYSCxDQVdOdkIsSUFYTTtBQVlkQyx1QkFBTyxDQUFDRCxLQUFELENBQVA7QUFDQUcsNEJBQVksQ0FBQyxLQUFELENBQVo7QUFiYztBQUFBOztBQUFBO0FBQUE7QUFBQTtBQWVkQSw0QkFBWSxDQUFDLEtBQUQsQ0FBWjtBQUNBSSx5QkFBUyxDQUFDLENBQUMsWUFBTWlCLFFBQU4sRUFBRCxDQUFELENBQVQ7O0FBQ0Esb0JBQUlaLDRDQUFLLENBQUNhLFFBQU4sYUFBSixFQUEyQjtBQUN6QnRCLDhCQUFZLENBQUMsS0FBRCxDQUFaO0FBQ0FJLDJCQUFTLENBQUMsQ0FBQyxpQkFBRCxDQUFELENBQVQ7QUFDRDs7QUFDRCwwQ0FBQUgsWUFBWSxDQUFDSyxPQUFiLGtGQUFzQkMsTUFBdEI7O0FBckJjO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQUg7O0FBQUEsc0JBQVRJLFNBQVM7QUFBQTtBQUFBO0FBQUEsT0FBZjs7QUF3QkEsUUFBSWhCLGVBQUosRUFBcUI7QUFDbkJnQixlQUFTO0FBQ1Y7O0FBQ0QsV0FBTztBQUFBLGFBQU1QLFNBQVMsQ0FBQyxFQUFELENBQWY7QUFBQSxLQUFQO0FBQ0QsR0FoQ1EsRUFnQ05WLFFBaENNLENBQVQ7QUFpQ0EsU0FBTztBQUFFRyxRQUFJLEVBQUpBLElBQUY7QUFBUUUsYUFBUyxFQUFUQSxTQUFSO0FBQW1CSSxVQUFNLEVBQU5BLE1BQW5CO0FBQTJCRixnQkFBWSxFQUFaQTtBQUEzQixHQUFQO0FBQ0QsQ0FuRE07O0dBQU1WLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMWNiZTRkN2NmOGRlNDk5YjkyMTcuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QsIHVzZVJlZiB9IGZyb20gJ3JlYWN0JztcbmltcG9ydCBheGlvcywge1xuICBBeGlvc1JlcXVlc3RDb25maWcsXG4gIEF4aW9zUmVzcG9uc2UsXG4gIENhbmNlbFRva2VuU291cmNlLFxufSBmcm9tICdheGlvcyc7XG5cbmltcG9ydCB7IHJvb3RfdXJsIH0gZnJvbSAnLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBnZXRQYXRoTmFtZSB9IGZyb20gJy4uL2NvbXBvbmVudHMvdXRpbHMnO1xuXG5pbnRlcmZhY2UgUmV0dXJuUmVxdWVzdCB7XG4gIGRhdGE6IGFueTtcbiAgZXJyb3JzOiBhbnlbXTtcbiAgaXNMb2FkaW5nOiBib29sZWFuO1xuICBjYW5jZWxTb3VyY2U6IGFueTtcbn1cblxuLy9mb3IgdHJhY2hpbmcsIHdoaWNoIHJlcS4gc2hvdWxkIGJlIGNhbmNlbGVkXG5leHBvcnQgY29uc3QgdXNlUmVxdWVzdCA9IChcbiAgdXJsOiBzdHJpbmcsXG4gIG9wdGlvbnM6IEF4aW9zUmVxdWVzdENvbmZpZyA9IHt9LFxuICB3YXRjaGVyczogYW55W10gPSBbXSxcbiAgc2hvdWxkX3dlX2ZldGNoOiBib29sZWFuID0gdHJ1ZVxuKTogUmV0dXJuUmVxdWVzdCA9PiB7XG4gIGNvbnN0IFtkYXRhLCBzZXREYXRhXSA9IHVzZVN0YXRlKG51bGwpO1xuICBjb25zdCBbaXNMb2FkaW5nLCBzZXRJc0xvYWRpbmddID0gdXNlU3RhdGUoZmFsc2UpO1xuICBjb25zdCBjYW5jZWxTb3VyY2UgPSB1c2VSZWY8Q2FuY2VsVG9rZW5Tb3VyY2UgfCBudWxsPihudWxsKTtcbiAgY29uc3QgW2Vycm9ycywgc2V0RXJyb3JzXSA9IHVzZVN0YXRlPHN0cmluZ1tdPihbXSk7XG5cbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZiAoY2FuY2VsU291cmNlKSB7XG4gICAgICBjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsKCk7XG4gICAgfVxuICB9LCBbXSk7XG5cbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBjb25zdCBDYW5jZWxUb2tlbiA9IGF4aW9zLkNhbmNlbFRva2VuO1xuICAgIGNhbmNlbFNvdXJjZS5jdXJyZW50ID0gQ2FuY2VsVG9rZW4uc291cmNlKCk7XG5cbiAgICBjb25zdCBmZXRjaERhdGEgPSBhc3luYyAoKSA9PiB7XG4gICAgICBhd2FpdCBzZXRJc0xvYWRpbmcodHJ1ZSk7XG4gICAgICB0cnkge1xuICAgICAgICBzZXRUaW1lb3V0KGNhbmNlbFNvdXJjZS5jdXJyZW50Py5jYW5jZWwsIDE4MDAwMCk7XG4gICAgICAgIGNvbnN0IGJhc2VQYXRoID0gZ2V0UGF0aE5hbWUoKVxuICAgICAgICBjb25zdCByZXNwb25zZTogQXhpb3NSZXNwb25zZSA9IGF3YWl0IGF4aW9zLnJlcXVlc3Qoe1xuICAgICAgICAgIHVybDogYCR7cm9vdF91cmx9JHt1cmx9YCxcbiAgICAgICAgICBtZXRob2Q6IG9wdGlvbnMubWV0aG9kIHx8ICdnZXQnLFxuICAgICAgICAgIGNhbmNlbFRva2VuOiBjYW5jZWxTb3VyY2UuY3VycmVudD8udG9rZW4sXG4gICAgICAgICAgLi4ub3B0aW9ucyxcbiAgICAgICAgfSk7XG4gICAgICAgIGNvbnN0IHsgZGF0YSB9ID0gcmVzcG9uc2U7XG4gICAgICAgIHNldERhdGEoZGF0YSk7XG4gICAgICAgIHNldElzTG9hZGluZyhmYWxzZSk7XG4gICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICBzZXRJc0xvYWRpbmcoZmFsc2UpO1xuICAgICAgICBzZXRFcnJvcnMoW2Vycm9yLnRvU3RyaW5nKCldKTtcbiAgICAgICAgaWYgKGF4aW9zLmlzQ2FuY2VsKGVycm9yKSkge1xuICAgICAgICAgIHNldElzTG9hZGluZyhmYWxzZSk7XG4gICAgICAgICAgc2V0RXJyb3JzKFsnUmVxdWVzdCBUaW1lb3V0J10pO1xuICAgICAgICB9XG4gICAgICAgIGNhbmNlbFNvdXJjZS5jdXJyZW50Py5jYW5jZWwoKTtcbiAgICAgIH1cbiAgICB9O1xuICAgIGlmIChzaG91bGRfd2VfZmV0Y2gpIHtcbiAgICAgIGZldGNoRGF0YSgpO1xuICAgIH1cbiAgICByZXR1cm4gKCkgPT4gc2V0RXJyb3JzKFtdKTtcbiAgfSwgd2F0Y2hlcnMpO1xuICByZXR1cm4geyBkYXRhLCBpc0xvYWRpbmcsIGVycm9ycywgY2FuY2VsU291cmNlIH07XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==
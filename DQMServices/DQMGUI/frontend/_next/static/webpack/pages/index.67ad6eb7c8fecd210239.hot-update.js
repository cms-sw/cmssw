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
                  url: "".concat(_config_config__WEBPACK_IMPORTED_MODULE_5__["root_url"]).concat(basePath).concat(url),
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlUmVxdWVzdC50c3giXSwibmFtZXMiOlsidXNlUmVxdWVzdCIsInVybCIsIm9wdGlvbnMiLCJ3YXRjaGVycyIsInNob3VsZF93ZV9mZXRjaCIsInVzZVN0YXRlIiwiZGF0YSIsInNldERhdGEiLCJpc0xvYWRpbmciLCJzZXRJc0xvYWRpbmciLCJjYW5jZWxTb3VyY2UiLCJ1c2VSZWYiLCJlcnJvcnMiLCJzZXRFcnJvcnMiLCJ1c2VFZmZlY3QiLCJjdXJyZW50IiwiY2FuY2VsIiwiQ2FuY2VsVG9rZW4iLCJheGlvcyIsInNvdXJjZSIsImZldGNoRGF0YSIsInNldFRpbWVvdXQiLCJiYXNlUGF0aCIsImdldFBhdGhOYW1lIiwicmVxdWVzdCIsInJvb3RfdXJsIiwibWV0aG9kIiwiY2FuY2VsVG9rZW4iLCJ0b2tlbiIsInJlc3BvbnNlIiwidG9TdHJpbmciLCJpc0NhbmNlbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBTUE7QUFDQTtBQVNBO0FBQ08sSUFBTUEsVUFBVSxHQUFHLFNBQWJBLFVBQWEsQ0FDeEJDLEdBRHdCLEVBS047QUFBQTs7QUFBQSxNQUhsQkMsT0FHa0IsdUVBSFksRUFHWjtBQUFBLE1BRmxCQyxRQUVrQix1RUFGQSxFQUVBO0FBQUEsTUFEbEJDLGVBQ2tCLHVFQURTLElBQ1Q7O0FBQUEsa0JBQ01DLHNEQUFRLENBQUMsSUFBRCxDQURkO0FBQUEsTUFDWEMsSUFEVztBQUFBLE1BQ0xDLE9BREs7O0FBQUEsbUJBRWdCRixzREFBUSxDQUFDLEtBQUQsQ0FGeEI7QUFBQSxNQUVYRyxTQUZXO0FBQUEsTUFFQUMsWUFGQTs7QUFHbEIsTUFBTUMsWUFBWSxHQUFHQyxvREFBTSxDQUEyQixJQUEzQixDQUEzQjs7QUFIa0IsbUJBSVVOLHNEQUFRLENBQVcsRUFBWCxDQUpsQjtBQUFBLE1BSVhPLE1BSlc7QUFBQSxNQUlIQyxTQUpHOztBQU1sQkMseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBSUosWUFBSixFQUFrQjtBQUFBOztBQUNoQiwrQkFBQUEsWUFBWSxDQUFDSyxPQUFiLGdGQUFzQkMsTUFBdEI7QUFDRDtBQUNGLEdBSlEsRUFJTixFQUpNLENBQVQ7QUFNQUYseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBTUcsV0FBVyxHQUFHQyw0Q0FBSyxDQUFDRCxXQUExQjtBQUNBUCxnQkFBWSxDQUFDSyxPQUFiLEdBQXVCRSxXQUFXLENBQUNFLE1BQVosRUFBdkI7O0FBRUEsUUFBTUMsU0FBUztBQUFBLGtNQUFHO0FBQUE7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLHVCQUNWWCxZQUFZLENBQUMsSUFBRCxDQURGOztBQUFBO0FBQUE7QUFHZFksMEJBQVUsMkJBQUNYLFlBQVksQ0FBQ0ssT0FBZCwyREFBQyx1QkFBc0JDLE1BQXZCLEVBQStCLE1BQS9CLENBQVY7QUFDTU0sd0JBSlEsR0FJR0MscUVBQVcsRUFKZDtBQUFBO0FBQUEsdUJBS3dCTCw0Q0FBSyxDQUFDTSxPQUFOO0FBQ3BDdkIscUJBQUcsWUFBS3dCLHVEQUFMLFNBQWdCSCxRQUFoQixTQUEyQnJCLEdBQTNCLENBRGlDO0FBRXBDeUIsd0JBQU0sRUFBRXhCLE9BQU8sQ0FBQ3dCLE1BQVIsSUFBa0IsS0FGVTtBQUdwQ0MsNkJBQVcsNEJBQUVqQixZQUFZLENBQUNLLE9BQWYsMkRBQUUsdUJBQXNCYTtBQUhDLG1CQUlqQzFCLE9BSmlDLEVBTHhCOztBQUFBO0FBS1IyQix3QkFMUTtBQVdOdkIscUJBWE0sR0FXR3VCLFFBWEgsQ0FXTnZCLElBWE07QUFZZEMsdUJBQU8sQ0FBQ0QsS0FBRCxDQUFQO0FBQ0FHLDRCQUFZLENBQUMsS0FBRCxDQUFaO0FBYmM7QUFBQTs7QUFBQTtBQUFBO0FBQUE7QUFlZEEsNEJBQVksQ0FBQyxLQUFELENBQVo7QUFDQUkseUJBQVMsQ0FBQyxDQUFDLFlBQU1pQixRQUFOLEVBQUQsQ0FBRCxDQUFUOztBQUNBLG9CQUFJWiw0Q0FBSyxDQUFDYSxRQUFOLGFBQUosRUFBMkI7QUFDekJ0Qiw4QkFBWSxDQUFDLEtBQUQsQ0FBWjtBQUNBSSwyQkFBUyxDQUFDLENBQUMsaUJBQUQsQ0FBRCxDQUFUO0FBQ0Q7O0FBQ0QsMENBQUFILFlBQVksQ0FBQ0ssT0FBYixrRkFBc0JDLE1BQXRCOztBQXJCYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFIOztBQUFBLHNCQUFUSSxTQUFTO0FBQUE7QUFBQTtBQUFBLE9BQWY7O0FBd0JBLFFBQUloQixlQUFKLEVBQXFCO0FBQ25CZ0IsZUFBUztBQUNWOztBQUNELFdBQU87QUFBQSxhQUFNUCxTQUFTLENBQUMsRUFBRCxDQUFmO0FBQUEsS0FBUDtBQUNELEdBaENRLEVBZ0NOVixRQWhDTSxDQUFUO0FBaUNBLFNBQU87QUFBRUcsUUFBSSxFQUFKQSxJQUFGO0FBQVFFLGFBQVMsRUFBVEEsU0FBUjtBQUFtQkksVUFBTSxFQUFOQSxNQUFuQjtBQUEyQkYsZ0JBQVksRUFBWkE7QUFBM0IsR0FBUDtBQUNELENBbkRNOztHQUFNVixVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjY3YWQ2ZWI3YzhmZWNkMjEwMjM5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0LCB1c2VSZWYgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgYXhpb3MsIHtcbiAgQXhpb3NSZXF1ZXN0Q29uZmlnLFxuICBBeGlvc1Jlc3BvbnNlLFxuICBDYW5jZWxUb2tlblNvdXJjZSxcbn0gZnJvbSAnYXhpb3MnO1xuXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgZ2V0UGF0aE5hbWUgfSBmcm9tICcuLi9jb21wb25lbnRzL3V0aWxzJztcblxuaW50ZXJmYWNlIFJldHVyblJlcXVlc3Qge1xuICBkYXRhOiBhbnk7XG4gIGVycm9yczogYW55W107XG4gIGlzTG9hZGluZzogYm9vbGVhbjtcbiAgY2FuY2VsU291cmNlOiBhbnk7XG59XG5cbi8vZm9yIHRyYWNoaW5nLCB3aGljaCByZXEuIHNob3VsZCBiZSBjYW5jZWxlZFxuZXhwb3J0IGNvbnN0IHVzZVJlcXVlc3QgPSAoXG4gIHVybDogc3RyaW5nLFxuICBvcHRpb25zOiBBeGlvc1JlcXVlc3RDb25maWcgPSB7fSxcbiAgd2F0Y2hlcnM6IGFueVtdID0gW10sXG4gIHNob3VsZF93ZV9mZXRjaDogYm9vbGVhbiA9IHRydWVcbik6IFJldHVyblJlcXVlc3QgPT4ge1xuICBjb25zdCBbZGF0YSwgc2V0RGF0YV0gPSB1c2VTdGF0ZShudWxsKTtcbiAgY29uc3QgW2lzTG9hZGluZywgc2V0SXNMb2FkaW5nXSA9IHVzZVN0YXRlKGZhbHNlKTtcbiAgY29uc3QgY2FuY2VsU291cmNlID0gdXNlUmVmPENhbmNlbFRva2VuU291cmNlIHwgbnVsbD4obnVsbCk7XG4gIGNvbnN0IFtlcnJvcnMsIHNldEVycm9yc10gPSB1c2VTdGF0ZTxzdHJpbmdbXT4oW10pO1xuXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgaWYgKGNhbmNlbFNvdXJjZSkge1xuICAgICAgY2FuY2VsU291cmNlLmN1cnJlbnQ/LmNhbmNlbCgpO1xuICAgIH1cbiAgfSwgW10pO1xuXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgY29uc3QgQ2FuY2VsVG9rZW4gPSBheGlvcy5DYW5jZWxUb2tlbjtcbiAgICBjYW5jZWxTb3VyY2UuY3VycmVudCA9IENhbmNlbFRva2VuLnNvdXJjZSgpO1xuXG4gICAgY29uc3QgZmV0Y2hEYXRhID0gYXN5bmMgKCkgPT4ge1xuICAgICAgYXdhaXQgc2V0SXNMb2FkaW5nKHRydWUpO1xuICAgICAgdHJ5IHtcbiAgICAgICAgc2V0VGltZW91dChjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsLCAxODAwMDApO1xuICAgICAgICBjb25zdCBiYXNlUGF0aCA9IGdldFBhdGhOYW1lKClcbiAgICAgICAgY29uc3QgcmVzcG9uc2U6IEF4aW9zUmVzcG9uc2UgPSBhd2FpdCBheGlvcy5yZXF1ZXN0KHtcbiAgICAgICAgICB1cmw6IGAke3Jvb3RfdXJsfSR7YmFzZVBhdGh9JHt1cmx9YCxcbiAgICAgICAgICBtZXRob2Q6IG9wdGlvbnMubWV0aG9kIHx8ICdnZXQnLFxuICAgICAgICAgIGNhbmNlbFRva2VuOiBjYW5jZWxTb3VyY2UuY3VycmVudD8udG9rZW4sXG4gICAgICAgICAgLi4ub3B0aW9ucyxcbiAgICAgICAgfSk7XG4gICAgICAgIGNvbnN0IHsgZGF0YSB9ID0gcmVzcG9uc2U7XG4gICAgICAgIHNldERhdGEoZGF0YSk7XG4gICAgICAgIHNldElzTG9hZGluZyhmYWxzZSk7XG4gICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICBzZXRJc0xvYWRpbmcoZmFsc2UpO1xuICAgICAgICBzZXRFcnJvcnMoW2Vycm9yLnRvU3RyaW5nKCldKTtcbiAgICAgICAgaWYgKGF4aW9zLmlzQ2FuY2VsKGVycm9yKSkge1xuICAgICAgICAgIHNldElzTG9hZGluZyhmYWxzZSk7XG4gICAgICAgICAgc2V0RXJyb3JzKFsnUmVxdWVzdCBUaW1lb3V0J10pO1xuICAgICAgICB9XG4gICAgICAgIGNhbmNlbFNvdXJjZS5jdXJyZW50Py5jYW5jZWwoKTtcbiAgICAgIH1cbiAgICB9O1xuICAgIGlmIChzaG91bGRfd2VfZmV0Y2gpIHtcbiAgICAgIGZldGNoRGF0YSgpO1xuICAgIH1cbiAgICByZXR1cm4gKCkgPT4gc2V0RXJyb3JzKFtdKTtcbiAgfSwgd2F0Y2hlcnMpO1xuICByZXR1cm4geyBkYXRhLCBpc0xvYWRpbmcsIGVycm9ycywgY2FuY2VsU291cmNlIH07XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==
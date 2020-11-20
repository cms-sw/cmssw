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
      setIsLoading = _useState2[1]; // const cancelSource = useRef<CancelTokenSource | null>(null);


  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])([]),
      errors = _useState3[0],
      setErrors = _useState3[1]; // useEffect(() => {
  //   if (cancelSource) {
  //     cancelSource.current?.cancel();
  //   }
  // }, []);


  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    var CancelToken = axios__WEBPACK_IMPORTED_MODULE_4___default.a.CancelToken; // cancelSource.current = CancelToken.source();

    var fetchData = /*#__PURE__*/function () {
      var _ref = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
        var response, _data;

        return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return setIsLoading(true);

              case 2:
                _context.prev = 2;
                _context.next = 5;
                return axios__WEBPACK_IMPORTED_MODULE_4___default.a.request(_objectSpread({
                  url: "".concat(_config_config__WEBPACK_IMPORTED_MODULE_5__["root_url"]).concat(url),
                  method: options.method || 'get'
                }, options));

              case 5:
                response = _context.sent;
                _data = response.data;
                setData(_data);
                setIsLoading(false);
                _context.next = 16;
                break;

              case 11:
                _context.prev = 11;
                _context.t0 = _context["catch"](2);
                setIsLoading(false);
                setErrors([_context.t0.toString()]);

                if (axios__WEBPACK_IMPORTED_MODULE_4___default.a.isCancel(_context.t0)) {
                  setIsLoading(false);
                  setErrors(['Request Timeout']);
                } // cancelSource.current?.cancel();


              case 16:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, null, [[2, 11]]);
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
  }, watchers); // return { data, isLoading, errors, cancelSource };

  return {
    data: data,
    isLoading: isLoading,
    errors: errors
  };
};

_s(useRequest, "KwcrWBun/28nifGaHwuhoB4o2uA=");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlUmVxdWVzdC50c3giXSwibmFtZXMiOlsidXNlUmVxdWVzdCIsInVybCIsIm9wdGlvbnMiLCJ3YXRjaGVycyIsInNob3VsZF93ZV9mZXRjaCIsInVzZVN0YXRlIiwiZGF0YSIsInNldERhdGEiLCJpc0xvYWRpbmciLCJzZXRJc0xvYWRpbmciLCJlcnJvcnMiLCJzZXRFcnJvcnMiLCJ1c2VFZmZlY3QiLCJDYW5jZWxUb2tlbiIsImF4aW9zIiwiZmV0Y2hEYXRhIiwicmVxdWVzdCIsInJvb3RfdXJsIiwibWV0aG9kIiwicmVzcG9uc2UiLCJ0b1N0cmluZyIsImlzQ2FuY2VsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQU1BO0FBVUE7QUFDTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxDQUN4QkMsR0FEd0IsRUFLTjtBQUFBOztBQUFBLE1BSGxCQyxPQUdrQix1RUFIWSxFQUdaO0FBQUEsTUFGbEJDLFFBRWtCLHVFQUZBLEVBRUE7QUFBQSxNQURsQkMsZUFDa0IsdUVBRFMsSUFDVDs7QUFBQSxrQkFDTUMsc0RBQVEsQ0FBQyxJQUFELENBRGQ7QUFBQSxNQUNYQyxJQURXO0FBQUEsTUFDTEMsT0FESzs7QUFBQSxtQkFFZ0JGLHNEQUFRLENBQUMsS0FBRCxDQUZ4QjtBQUFBLE1BRVhHLFNBRlc7QUFBQSxNQUVBQyxZQUZBLGtCQUdsQjs7O0FBSGtCLG1CQUlVSixzREFBUSxDQUFXLEVBQVgsQ0FKbEI7QUFBQSxNQUlYSyxNQUpXO0FBQUEsTUFJSEMsU0FKRyxrQkFNbEI7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7O0FBRUFDLHlEQUFTLENBQUMsWUFBTTtBQUNkLFFBQU1DLFdBQVcsR0FBR0MsNENBQUssQ0FBQ0QsV0FBMUIsQ0FEYyxDQUVkOztBQUVBLFFBQU1FLFNBQVM7QUFBQSxrTUFBRztBQUFBOztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSx1QkFDVk4sWUFBWSxDQUFDLElBQUQsQ0FERjs7QUFBQTtBQUFBO0FBQUE7QUFBQSx1QkFJd0JLLDRDQUFLLENBQUNFLE9BQU47QUFDcENmLHFCQUFHLFlBQUtnQix1REFBTCxTQUFnQmhCLEdBQWhCLENBRGlDO0FBRXBDaUIsd0JBQU0sRUFBRWhCLE9BQU8sQ0FBQ2dCLE1BQVIsSUFBa0I7QUFGVSxtQkFJakNoQixPQUppQyxFQUp4Qjs7QUFBQTtBQUlSaUIsd0JBSlE7QUFVTmIscUJBVk0sR0FVR2EsUUFWSCxDQVVOYixJQVZNO0FBV2RDLHVCQUFPLENBQUNELEtBQUQsQ0FBUDtBQUNBRyw0QkFBWSxDQUFDLEtBQUQsQ0FBWjtBQVpjO0FBQUE7O0FBQUE7QUFBQTtBQUFBO0FBY2RBLDRCQUFZLENBQUMsS0FBRCxDQUFaO0FBQ0FFLHlCQUFTLENBQUMsQ0FBQyxZQUFNUyxRQUFOLEVBQUQsQ0FBRCxDQUFUOztBQUNBLG9CQUFJTiw0Q0FBSyxDQUFDTyxRQUFOLGFBQUosRUFBMkI7QUFDekJaLDhCQUFZLENBQUMsS0FBRCxDQUFaO0FBQ0FFLDJCQUFTLENBQUMsQ0FBQyxpQkFBRCxDQUFELENBQVQ7QUFDRCxpQkFuQmEsQ0FvQmQ7OztBQXBCYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFIOztBQUFBLHNCQUFUSSxTQUFTO0FBQUE7QUFBQTtBQUFBLE9BQWY7O0FBdUJBLFFBQUlYLGVBQUosRUFBcUI7QUFDbkJXLGVBQVM7QUFDVjs7QUFDRCxXQUFPO0FBQUEsYUFBTUosU0FBUyxDQUFDLEVBQUQsQ0FBZjtBQUFBLEtBQVA7QUFDRCxHQS9CUSxFQStCTlIsUUEvQk0sQ0FBVCxDQVprQixDQTRDbEI7O0FBRUEsU0FBTztBQUFFRyxRQUFJLEVBQUpBLElBQUY7QUFBUUUsYUFBUyxFQUFUQSxTQUFSO0FBQW1CRSxVQUFNLEVBQU5BO0FBQW5CLEdBQVA7QUFDRCxDQXBETTs7R0FBTVYsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4wNTcyZmQ4NjA5ZDM0Mzg4YTYyZC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgdXNlU3RhdGUsIHVzZUVmZmVjdCwgdXNlUmVmIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IGF4aW9zLCB7XG4gIEF4aW9zUmVxdWVzdENvbmZpZyxcbiAgQXhpb3NSZXNwb25zZSxcbiAgQ2FuY2VsVG9rZW5Tb3VyY2UsXG59IGZyb20gJ2F4aW9zJztcblxuaW1wb3J0IHsgcm9vdF91cmwgfSBmcm9tICcuLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IGdldFBhdGhOYW1lIH0gZnJvbSAnLi4vY29tcG9uZW50cy91dGlscyc7XG5cbmludGVyZmFjZSBSZXR1cm5SZXF1ZXN0IHtcbiAgZGF0YTogYW55O1xuICBlcnJvcnM6IGFueVtdO1xuICBpc0xvYWRpbmc6IGJvb2xlYW47XG4gIGNhbmNlbFNvdXJjZTogYW55O1xufVxuXG4vL2ZvciB0cmFjaGluZywgd2hpY2ggcmVxLiBzaG91bGQgYmUgY2FuY2VsZWRcbmV4cG9ydCBjb25zdCB1c2VSZXF1ZXN0ID0gKFxuICB1cmw6IHN0cmluZyxcbiAgb3B0aW9uczogQXhpb3NSZXF1ZXN0Q29uZmlnID0ge30sXG4gIHdhdGNoZXJzOiBhbnlbXSA9IFtdLFxuICBzaG91bGRfd2VfZmV0Y2g6IGJvb2xlYW4gPSB0cnVlXG4pOiBSZXR1cm5SZXF1ZXN0ID0+IHtcbiAgY29uc3QgW2RhdGEsIHNldERhdGFdID0gdXNlU3RhdGUobnVsbCk7XG4gIGNvbnN0IFtpc0xvYWRpbmcsIHNldElzTG9hZGluZ10gPSB1c2VTdGF0ZShmYWxzZSk7XG4gIC8vIGNvbnN0IGNhbmNlbFNvdXJjZSA9IHVzZVJlZjxDYW5jZWxUb2tlblNvdXJjZSB8IG51bGw+KG51bGwpO1xuICBjb25zdCBbZXJyb3JzLCBzZXRFcnJvcnNdID0gdXNlU3RhdGU8c3RyaW5nW10+KFtdKTtcblxuICAvLyB1c2VFZmZlY3QoKCkgPT4ge1xuICAvLyAgIGlmIChjYW5jZWxTb3VyY2UpIHtcbiAgLy8gICAgIGNhbmNlbFNvdXJjZS5jdXJyZW50Py5jYW5jZWwoKTtcbiAgLy8gICB9XG4gIC8vIH0sIFtdKTtcblxuICB1c2VFZmZlY3QoKCkgPT4ge1xuICAgIGNvbnN0IENhbmNlbFRva2VuID0gYXhpb3MuQ2FuY2VsVG9rZW47XG4gICAgLy8gY2FuY2VsU291cmNlLmN1cnJlbnQgPSBDYW5jZWxUb2tlbi5zb3VyY2UoKTtcblxuICAgIGNvbnN0IGZldGNoRGF0YSA9IGFzeW5jICgpID0+IHtcbiAgICAgIGF3YWl0IHNldElzTG9hZGluZyh0cnVlKTtcbiAgICAgIHRyeSB7XG4gICAgICAgIC8vIHNldFRpbWVvdXQoY2FuY2VsU291cmNlLmN1cnJlbnQ/LmNhbmNlbCwgMTgwMDAwKTtcbiAgICAgICAgY29uc3QgcmVzcG9uc2U6IEF4aW9zUmVzcG9uc2UgPSBhd2FpdCBheGlvcy5yZXF1ZXN0KHtcbiAgICAgICAgICB1cmw6IGAke3Jvb3RfdXJsfSR7dXJsfWAsXG4gICAgICAgICAgbWV0aG9kOiBvcHRpb25zLm1ldGhvZCB8fCAnZ2V0JyxcbiAgICAgICAgICAvLyBjYW5jZWxUb2tlbjogY2FuY2VsU291cmNlLmN1cnJlbnQ/LnRva2VuLFxuICAgICAgICAgIC4uLm9wdGlvbnMsXG4gICAgICAgIH0pO1xuICAgICAgICBjb25zdCB7IGRhdGEgfSA9IHJlc3BvbnNlO1xuICAgICAgICBzZXREYXRhKGRhdGEpO1xuICAgICAgICBzZXRJc0xvYWRpbmcoZmFsc2UpO1xuICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgc2V0SXNMb2FkaW5nKGZhbHNlKTtcbiAgICAgICAgc2V0RXJyb3JzKFtlcnJvci50b1N0cmluZygpXSk7XG4gICAgICAgIGlmIChheGlvcy5pc0NhbmNlbChlcnJvcikpIHtcbiAgICAgICAgICBzZXRJc0xvYWRpbmcoZmFsc2UpO1xuICAgICAgICAgIHNldEVycm9ycyhbJ1JlcXVlc3QgVGltZW91dCddKTtcbiAgICAgICAgfVxuICAgICAgICAvLyBjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsKCk7XG4gICAgICB9XG4gICAgfTtcbiAgICBpZiAoc2hvdWxkX3dlX2ZldGNoKSB7XG4gICAgICBmZXRjaERhdGEoKTtcbiAgICB9XG4gICAgcmV0dXJuICgpID0+IHNldEVycm9ycyhbXSk7XG4gIH0sIHdhdGNoZXJzKTtcbiAgLy8gcmV0dXJuIHsgZGF0YSwgaXNMb2FkaW5nLCBlcnJvcnMsIGNhbmNlbFNvdXJjZSB9O1xuXG4gIHJldHVybiB7IGRhdGEsIGlzTG9hZGluZywgZXJyb3JzIH07XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==
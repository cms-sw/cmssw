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
        var _cancelSource$current2, _cancelSource$current3, response, _data, _cancelSource$current4;

        return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return setIsLoading(true);

              case 2:
                _context.prev = 2;
                setTimeout((_cancelSource$current2 = cancelSource.current) === null || _cancelSource$current2 === void 0 ? void 0 : _cancelSource$current2.cancel, 180000);
                _context.next = 6;
                return axios__WEBPACK_IMPORTED_MODULE_4___default.a.request(_objectSpread({
                  url: "".concat(_config_config__WEBPACK_IMPORTED_MODULE_5__["root_url"]).concat(url),
                  method: options.method || 'get',
                  cancelToken: (_cancelSource$current3 = cancelSource.current) === null || _cancelSource$current3 === void 0 ? void 0 : _cancelSource$current3.token
                }, options));

              case 6:
                response = _context.sent;
                _data = response.data;
                setData(_data);
                setIsLoading(false);
                _context.next = 18;
                break;

              case 12:
                _context.prev = 12;
                _context.t0 = _context["catch"](2);
                setIsLoading(false);
                setErrors([_context.t0.toString()]);

                if (axios__WEBPACK_IMPORTED_MODULE_4___default.a.isCancel(_context.t0)) {
                  setIsLoading(false);
                  setErrors(['Request Timeout']);
                }

                (_cancelSource$current4 = cancelSource.current) === null || _cancelSource$current4 === void 0 ? void 0 : _cancelSource$current4.cancel();

              case 18:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, null, [[2, 12]]);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlUmVxdWVzdC50c3giXSwibmFtZXMiOlsidXNlUmVxdWVzdCIsInVybCIsIm9wdGlvbnMiLCJ3YXRjaGVycyIsInNob3VsZF93ZV9mZXRjaCIsInVzZVN0YXRlIiwiZGF0YSIsInNldERhdGEiLCJpc0xvYWRpbmciLCJzZXRJc0xvYWRpbmciLCJjYW5jZWxTb3VyY2UiLCJ1c2VSZWYiLCJlcnJvcnMiLCJzZXRFcnJvcnMiLCJ1c2VFZmZlY3QiLCJjdXJyZW50IiwiY2FuY2VsIiwiQ2FuY2VsVG9rZW4iLCJheGlvcyIsInNvdXJjZSIsImZldGNoRGF0YSIsInNldFRpbWVvdXQiLCJyZXF1ZXN0Iiwicm9vdF91cmwiLCJtZXRob2QiLCJjYW5jZWxUb2tlbiIsInRva2VuIiwicmVzcG9uc2UiLCJ0b1N0cmluZyIsImlzQ2FuY2VsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQU1BO0FBU0E7QUFDTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxDQUN4QkMsR0FEd0IsRUFLTjtBQUFBOztBQUFBLE1BSGxCQyxPQUdrQix1RUFIWSxFQUdaO0FBQUEsTUFGbEJDLFFBRWtCLHVFQUZBLEVBRUE7QUFBQSxNQURsQkMsZUFDa0IsdUVBRFMsSUFDVDs7QUFBQSxrQkFDTUMsc0RBQVEsQ0FBQyxJQUFELENBRGQ7QUFBQSxNQUNYQyxJQURXO0FBQUEsTUFDTEMsT0FESzs7QUFBQSxtQkFFZ0JGLHNEQUFRLENBQUMsS0FBRCxDQUZ4QjtBQUFBLE1BRVhHLFNBRlc7QUFBQSxNQUVBQyxZQUZBOztBQUdsQixNQUFNQyxZQUFZLEdBQUdDLG9EQUFNLENBQTJCLElBQTNCLENBQTNCOztBQUhrQixtQkFJVU4sc0RBQVEsQ0FBVyxFQUFYLENBSmxCO0FBQUEsTUFJWE8sTUFKVztBQUFBLE1BSUhDLFNBSkc7O0FBTWxCQyx5REFBUyxDQUFDLFlBQU07QUFDZCxRQUFJSixZQUFKLEVBQWtCO0FBQUE7O0FBQ2hCLCtCQUFBQSxZQUFZLENBQUNLLE9BQWIsZ0ZBQXNCQyxNQUF0QjtBQUNEO0FBQ0YsR0FKUSxFQUlOLEVBSk0sQ0FBVDtBQU1BRix5REFBUyxDQUFDLFlBQU07QUFDZCxRQUFNRyxXQUFXLEdBQUdDLDRDQUFLLENBQUNELFdBQTFCO0FBQ0FQLGdCQUFZLENBQUNLLE9BQWIsR0FBdUJFLFdBQVcsQ0FBQ0UsTUFBWixFQUF2Qjs7QUFFQSxRQUFNQyxTQUFTO0FBQUEsa01BQUc7QUFBQTs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsdUJBQ1ZYLFlBQVksQ0FBQyxJQUFELENBREY7O0FBQUE7QUFBQTtBQUdkWSwwQkFBVSwyQkFBQ1gsWUFBWSxDQUFDSyxPQUFkLDJEQUFDLHVCQUFzQkMsTUFBdkIsRUFBK0IsTUFBL0IsQ0FBVjtBQUhjO0FBQUEsdUJBSXdCRSw0Q0FBSyxDQUFDSSxPQUFOO0FBQ3BDckIscUJBQUcsWUFBS3NCLHVEQUFMLFNBQWdCdEIsR0FBaEIsQ0FEaUM7QUFFcEN1Qix3QkFBTSxFQUFFdEIsT0FBTyxDQUFDc0IsTUFBUixJQUFrQixLQUZVO0FBR3BDQyw2QkFBVyw0QkFBRWYsWUFBWSxDQUFDSyxPQUFmLDJEQUFFLHVCQUFzQlc7QUFIQyxtQkFJakN4QixPQUppQyxFQUp4Qjs7QUFBQTtBQUlSeUIsd0JBSlE7QUFVTnJCLHFCQVZNLEdBVUdxQixRQVZILENBVU5yQixJQVZNO0FBV2RDLHVCQUFPLENBQUNELEtBQUQsQ0FBUDtBQUNBRyw0QkFBWSxDQUFDLEtBQUQsQ0FBWjtBQVpjO0FBQUE7O0FBQUE7QUFBQTtBQUFBO0FBY2RBLDRCQUFZLENBQUMsS0FBRCxDQUFaO0FBQ0FJLHlCQUFTLENBQUMsQ0FBQyxZQUFNZSxRQUFOLEVBQUQsQ0FBRCxDQUFUOztBQUNBLG9CQUFJViw0Q0FBSyxDQUFDVyxRQUFOLGFBQUosRUFBMkI7QUFDekJwQiw4QkFBWSxDQUFDLEtBQUQsQ0FBWjtBQUNBSSwyQkFBUyxDQUFDLENBQUMsaUJBQUQsQ0FBRCxDQUFUO0FBQ0Q7O0FBQ0QsMENBQUFILFlBQVksQ0FBQ0ssT0FBYixrRkFBc0JDLE1BQXRCOztBQXBCYztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFIOztBQUFBLHNCQUFUSSxTQUFTO0FBQUE7QUFBQTtBQUFBLE9BQWY7O0FBdUJBLFFBQUloQixlQUFKLEVBQXFCO0FBQ25CZ0IsZUFBUztBQUNWOztBQUNELFdBQU87QUFBQSxhQUFNUCxTQUFTLENBQUMsRUFBRCxDQUFmO0FBQUEsS0FBUDtBQUNELEdBL0JRLEVBK0JOVixRQS9CTSxDQUFUO0FBZ0NBLFNBQU87QUFBRUcsUUFBSSxFQUFKQSxJQUFGO0FBQVFFLGFBQVMsRUFBVEEsU0FBUjtBQUFtQkksVUFBTSxFQUFOQSxNQUFuQjtBQUEyQkYsZ0JBQVksRUFBWkE7QUFBM0IsR0FBUDtBQUNELENBbERNOztHQUFNVixVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmE3ZDJiOWUxMDUxODlmNGM3YzE3LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0LCB1c2VSZWYgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCBheGlvcywge1xyXG4gIEF4aW9zUmVxdWVzdENvbmZpZyxcclxuICBBeGlvc1Jlc3BvbnNlLFxyXG4gIENhbmNlbFRva2VuU291cmNlLFxyXG59IGZyb20gJ2F4aW9zJztcclxuXHJcbmltcG9ydCB7IHJvb3RfdXJsIH0gZnJvbSAnLi4vY29uZmlnL2NvbmZpZyc7XHJcblxyXG5pbnRlcmZhY2UgUmV0dXJuUmVxdWVzdCB7XHJcbiAgZGF0YTogYW55O1xyXG4gIGVycm9yczogYW55W107XHJcbiAgaXNMb2FkaW5nOiBib29sZWFuO1xyXG4gIGNhbmNlbFNvdXJjZTogYW55O1xyXG59XHJcblxyXG4vL2ZvciB0cmFjaGluZywgd2hpY2ggcmVxLiBzaG91bGQgYmUgY2FuY2VsZWRcclxuZXhwb3J0IGNvbnN0IHVzZVJlcXVlc3QgPSAoXHJcbiAgdXJsOiBzdHJpbmcsXHJcbiAgb3B0aW9uczogQXhpb3NSZXF1ZXN0Q29uZmlnID0ge30sXHJcbiAgd2F0Y2hlcnM6IGFueVtdID0gW10sXHJcbiAgc2hvdWxkX3dlX2ZldGNoOiBib29sZWFuID0gdHJ1ZVxyXG4pOiBSZXR1cm5SZXF1ZXN0ID0+IHtcclxuICBjb25zdCBbZGF0YSwgc2V0RGF0YV0gPSB1c2VTdGF0ZShudWxsKTtcclxuICBjb25zdCBbaXNMb2FkaW5nLCBzZXRJc0xvYWRpbmddID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IGNhbmNlbFNvdXJjZSA9IHVzZVJlZjxDYW5jZWxUb2tlblNvdXJjZSB8IG51bGw+KG51bGwpO1xyXG4gIGNvbnN0IFtlcnJvcnMsIHNldEVycm9yc10gPSB1c2VTdGF0ZTxzdHJpbmdbXT4oW10pO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKGNhbmNlbFNvdXJjZSkge1xyXG4gICAgICBjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsKCk7XHJcbiAgICB9XHJcbiAgfSwgW10pO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgQ2FuY2VsVG9rZW4gPSBheGlvcy5DYW5jZWxUb2tlbjtcclxuICAgIGNhbmNlbFNvdXJjZS5jdXJyZW50ID0gQ2FuY2VsVG9rZW4uc291cmNlKCk7XHJcblxyXG4gICAgY29uc3QgZmV0Y2hEYXRhID0gYXN5bmMgKCkgPT4ge1xyXG4gICAgICBhd2FpdCBzZXRJc0xvYWRpbmcodHJ1ZSk7XHJcbiAgICAgIHRyeSB7XHJcbiAgICAgICAgc2V0VGltZW91dChjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsLCAxODAwMDApO1xyXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlOiBBeGlvc1Jlc3BvbnNlID0gYXdhaXQgYXhpb3MucmVxdWVzdCh7XHJcbiAgICAgICAgICB1cmw6IGAke3Jvb3RfdXJsfSR7dXJsfWAsXHJcbiAgICAgICAgICBtZXRob2Q6IG9wdGlvbnMubWV0aG9kIHx8ICdnZXQnLFxyXG4gICAgICAgICAgY2FuY2VsVG9rZW46IGNhbmNlbFNvdXJjZS5jdXJyZW50Py50b2tlbixcclxuICAgICAgICAgIC4uLm9wdGlvbnMsXHJcbiAgICAgICAgfSk7XHJcbiAgICAgICAgY29uc3QgeyBkYXRhIH0gPSByZXNwb25zZTtcclxuICAgICAgICBzZXREYXRhKGRhdGEpO1xyXG4gICAgICAgIHNldElzTG9hZGluZyhmYWxzZSk7XHJcbiAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgc2V0SXNMb2FkaW5nKGZhbHNlKTtcclxuICAgICAgICBzZXRFcnJvcnMoW2Vycm9yLnRvU3RyaW5nKCldKTtcclxuICAgICAgICBpZiAoYXhpb3MuaXNDYW5jZWwoZXJyb3IpKSB7XHJcbiAgICAgICAgICBzZXRJc0xvYWRpbmcoZmFsc2UpO1xyXG4gICAgICAgICAgc2V0RXJyb3JzKFsnUmVxdWVzdCBUaW1lb3V0J10pO1xyXG4gICAgICAgIH1cclxuICAgICAgICBjYW5jZWxTb3VyY2UuY3VycmVudD8uY2FuY2VsKCk7XHJcbiAgICAgIH1cclxuICAgIH07XHJcbiAgICBpZiAoc2hvdWxkX3dlX2ZldGNoKSB7XHJcbiAgICAgIGZldGNoRGF0YSgpO1xyXG4gICAgfVxyXG4gICAgcmV0dXJuICgpID0+IHNldEVycm9ycyhbXSk7XHJcbiAgfSwgd2F0Y2hlcnMpO1xyXG4gIHJldHVybiB7IGRhdGEsIGlzTG9hZGluZywgZXJyb3JzLCBjYW5jZWxTb3VyY2UgfTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
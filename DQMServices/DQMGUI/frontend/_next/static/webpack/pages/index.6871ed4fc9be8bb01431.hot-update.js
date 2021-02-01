webpackHotUpdate_N_E("pages/index",{

/***/ "./components/liveModeButton.tsx":
/*!***************************************!*\
  !*** ./components/liveModeButton.tsx ***!
  \***************************************/
/*! exports provided: LiveModeButton */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LiveModeButton", function() { return LiveModeButton; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/liveModeButton.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var liveModeHandler = function liveModeHandler(liveModeRun, liveModeDataset) {
  next_router__WEBPACK_IMPORTED_MODULE_2___default.a.push({
    pathname: '/',
    query: {
      run_number: liveModeRun,
      dataset_name: liveModeDataset,
      folder_path: 'Summary'
    }
  });
};

var LiveModeButton = function LiveModeButton() {
  _s();

  var liveModeDataset = '/Global/Online/ALL';
  var liveModeRun = '0';

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__["useUpdateLiveMode"])(),
      set_update = _useUpdateLiveMode.set_update;

  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_1__["LiveButton"], {
    onClick: function onClick() {
      liveModeHandler(liveModeRun, liveModeDataset);
      set_update(true);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 24,
      columnNumber: 5
    }
  }, "Live Mode");
};

_s(LiveModeButton, "DzlPWT6eG2dqwLqJrtagWcQ6sa0=", false, function () {
  return [_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__["useUpdateLiveMode"]];
});

_c = LiveModeButton;

var _c;

$RefreshReg$(_c, "LiveModeButton");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9saXZlTW9kZUJ1dHRvbi50c3giXSwibmFtZXMiOlsibGl2ZU1vZGVIYW5kbGVyIiwibGl2ZU1vZGVSdW4iLCJsaXZlTW9kZURhdGFzZXQiLCJSb3V0ZXIiLCJwdXNoIiwicGF0aG5hbWUiLCJxdWVyeSIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJmb2xkZXJfcGF0aCIsIkxpdmVNb2RlQnV0dG9uIiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJzZXRfdXBkYXRlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUNBOztBQUVBLElBQU1BLGVBQWUsR0FBRyxTQUFsQkEsZUFBa0IsQ0FBQ0MsV0FBRCxFQUFzQkMsZUFBdEIsRUFBa0Q7QUFDeEVDLG9EQUFNLENBQUNDLElBQVAsQ0FBWTtBQUNWQyxZQUFRLEVBQUUsR0FEQTtBQUVWQyxTQUFLLEVBQUU7QUFDTEMsZ0JBQVUsRUFBRU4sV0FEUDtBQUVMTyxrQkFBWSxFQUFFTixlQUZUO0FBR0xPLGlCQUFXLEVBQUU7QUFIUjtBQUZHLEdBQVo7QUFRRCxDQVREOztBQVdPLElBQU1DLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsR0FBTTtBQUFBOztBQUNsQyxNQUFNUixlQUFlLEdBQUcsb0JBQXhCO0FBQ0EsTUFBTUQsV0FBVyxHQUFHLEdBQXBCOztBQUZrQywyQkFHWFUsb0ZBQWlCLEVBSE47QUFBQSxNQUcxQkMsVUFIMEIsc0JBRzFCQSxVQUgwQjs7QUFLbEMsU0FDRSxNQUFDLDREQUFEO0FBQ0UsV0FBTyxFQUFFLG1CQUFNO0FBQ2JaLHFCQUFlLENBQUNDLFdBQUQsRUFBY0MsZUFBZCxDQUFmO0FBQ0FVLGdCQUFVLENBQUMsSUFBRCxDQUFWO0FBQ0QsS0FKSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGlCQURGO0FBVUQsQ0FmTTs7R0FBTUYsYztVQUdZQyw0RTs7O0tBSFpELGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNjg3MWVkNGZjOWJlOGJiMDE0MzEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuXHJcbmltcG9ydCB7IExpdmVCdXR0b24gfSBmcm9tICcuL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcclxuXHJcbmNvbnN0IGxpdmVNb2RlSGFuZGxlciA9IChsaXZlTW9kZVJ1bjogc3RyaW5nLCBsaXZlTW9kZURhdGFzZXQ6IHN0cmluZykgPT4ge1xyXG4gIFJvdXRlci5wdXNoKHtcclxuICAgIHBhdGhuYW1lOiAnLycsXHJcbiAgICBxdWVyeToge1xyXG4gICAgICBydW5fbnVtYmVyOiBsaXZlTW9kZVJ1bixcclxuICAgICAgZGF0YXNldF9uYW1lOiBsaXZlTW9kZURhdGFzZXQsXHJcbiAgICAgIGZvbGRlcl9wYXRoOiAnU3VtbWFyeScsXHJcbiAgICB9LFxyXG4gIH0pO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IExpdmVNb2RlQnV0dG9uID0gKCkgPT4ge1xyXG4gIGNvbnN0IGxpdmVNb2RlRGF0YXNldCA9ICcvR2xvYmFsL09ubGluZS9BTEwnO1xyXG4gIGNvbnN0IGxpdmVNb2RlUnVuID0gJzAnO1xyXG4gIGNvbnN0IHsgc2V0X3VwZGF0ZSB9ID0gdXNlVXBkYXRlTGl2ZU1vZGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxMaXZlQnV0dG9uXHJcbiAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICBsaXZlTW9kZUhhbmRsZXIobGl2ZU1vZGVSdW4sIGxpdmVNb2RlRGF0YXNldCk7XHJcbiAgICAgICAgc2V0X3VwZGF0ZSh0cnVlKTtcclxuICAgICAgfX1cclxuICAgID5cclxuICAgICAgTGl2ZSBNb2RlXHJcbiAgICA8L0xpdmVCdXR0b24+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
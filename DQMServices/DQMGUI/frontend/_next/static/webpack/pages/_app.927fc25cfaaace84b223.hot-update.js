webpackHotUpdate_N_E("pages/_app",{

/***/ "./styles/theme.ts":
/*!*************************!*\
  !*** ./styles/theme.ts ***!
  \*************************/
/*! exports provided: theme */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "theme", function() { return theme; });
var theme = {
  colors: {
    primary: {
      main: 'green',
      light: '#C5CBE3'
    },
    secondary: {
      main: '#AC3B61',
      light: '#edc7b7',
      dark: '#5d1d32',
      darkContrast: '#fff'
    },
    thirdy: {
      dark: '#4C6B9D',
      light: '#A4ABBD'
    },
    common: {
      white: '#fff',
      black: '#000',
      lightGrey: '#dfdfdf',
      blueGrey: '#f0f2f5'
    },
    notification: {
      error: '#d01f1f',
      success: '#41c12d',
      darkSuccess: '#3eb02c',
      warning: '#f08e13'
    }
  },
  fontFamily: {
    sansSerif: '-apple-system, "Helvetica Neue", Arial, sans-serif',
    mono: 'Menlo, Monaco, monospace'
  },
  space: {
    padding: '4px',
    spaceBetween: '4px'
  }
};

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3R5bGVzL3RoZW1lLnRzIl0sIm5hbWVzIjpbInRoZW1lIiwiY29sb3JzIiwicHJpbWFyeSIsIm1haW4iLCJsaWdodCIsInNlY29uZGFyeSIsImRhcmsiLCJkYXJrQ29udHJhc3QiLCJ0aGlyZHkiLCJjb21tb24iLCJ3aGl0ZSIsImJsYWNrIiwibGlnaHRHcmV5IiwiYmx1ZUdyZXkiLCJub3RpZmljYXRpb24iLCJlcnJvciIsInN1Y2Nlc3MiLCJkYXJrU3VjY2VzcyIsIndhcm5pbmciLCJmb250RmFtaWx5Iiwic2Fuc1NlcmlmIiwibW9ubyIsInNwYWNlIiwicGFkZGluZyIsInNwYWNlQmV0d2VlbiJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBTyxJQUFNQSxLQUFLLEdBQUc7QUFDbkJDLFFBQU0sRUFBRTtBQUNOQyxXQUFPLEVBQUU7QUFDUEMsVUFBSSxFQUFFLE9BREM7QUFFUEMsV0FBSyxFQUFFO0FBRkEsS0FESDtBQUtOQyxhQUFTLEVBQUU7QUFDVEYsVUFBSSxFQUFFLFNBREc7QUFFVEMsV0FBSyxFQUFFLFNBRkU7QUFHVEUsVUFBSSxFQUFFLFNBSEc7QUFJVEMsa0JBQVksRUFBRTtBQUpMLEtBTEw7QUFXTkMsVUFBTSxFQUFFO0FBQ05GLFVBQUksRUFBRSxTQURBO0FBRU5GLFdBQUssRUFBRTtBQUZELEtBWEY7QUFlTkssVUFBTSxFQUFFO0FBQ05DLFdBQUssRUFBRSxNQUREO0FBRU5DLFdBQUssRUFBRSxNQUZEO0FBR05DLGVBQVMsRUFBRSxTQUhMO0FBSU5DLGNBQVEsRUFBRTtBQUpKLEtBZkY7QUFxQk5DLGdCQUFZLEVBQUU7QUFDWkMsV0FBSyxFQUFFLFNBREs7QUFFWkMsYUFBTyxFQUFFLFNBRkc7QUFHWkMsaUJBQVcsRUFBRSxTQUhEO0FBSVpDLGFBQU8sRUFBRTtBQUpHO0FBckJSLEdBRFc7QUE2Qm5CQyxZQUFVLEVBQUU7QUFDVkMsYUFBUyxFQUFFLG9EQUREO0FBRVZDLFFBQUksRUFBRTtBQUZJLEdBN0JPO0FBaUNuQkMsT0FBSyxFQUFFO0FBQ0xDLFdBQU8sRUFBRSxLQURKO0FBRUxDLGdCQUFZLEVBQUU7QUFGVDtBQWpDWSxDQUFkIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL19hcHAuOTI3ZmMyNWNmYWFhY2U4NGIyMjMuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImV4cG9ydCBjb25zdCB0aGVtZSA9IHtcclxuICBjb2xvcnM6IHtcclxuICAgIHByaW1hcnk6IHtcclxuICAgICAgbWFpbjogJ2dyZWVuJyxcclxuICAgICAgbGlnaHQ6ICcjQzVDQkUzJyxcclxuICAgIH0sXHJcbiAgICBzZWNvbmRhcnk6IHtcclxuICAgICAgbWFpbjogJyNBQzNCNjEnLFxyXG4gICAgICBsaWdodDogJyNlZGM3YjcnLFxyXG4gICAgICBkYXJrOiAnIzVkMWQzMicsXHJcbiAgICAgIGRhcmtDb250cmFzdDogJyNmZmYnLFxyXG4gICAgfSxcclxuICAgIHRoaXJkeToge1xyXG4gICAgICBkYXJrOiAnIzRDNkI5RCcsXHJcbiAgICAgIGxpZ2h0OiAnI0E0QUJCRCcsXHJcbiAgICB9LFxyXG4gICAgY29tbW9uOiB7XHJcbiAgICAgIHdoaXRlOiAnI2ZmZicsXHJcbiAgICAgIGJsYWNrOiAnIzAwMCcsXHJcbiAgICAgIGxpZ2h0R3JleTogJyNkZmRmZGYnLFxyXG4gICAgICBibHVlR3JleTogJyNmMGYyZjUnLFxyXG4gICAgfSxcclxuICAgIG5vdGlmaWNhdGlvbjoge1xyXG4gICAgICBlcnJvcjogJyNkMDFmMWYnLFxyXG4gICAgICBzdWNjZXNzOiAnIzQxYzEyZCcsXHJcbiAgICAgIGRhcmtTdWNjZXNzOiAnIzNlYjAyYycsXHJcbiAgICAgIHdhcm5pbmc6ICcjZjA4ZTEzJyxcclxuICAgIH0sXHJcbiAgfSxcclxuICBmb250RmFtaWx5OiB7XHJcbiAgICBzYW5zU2VyaWY6ICctYXBwbGUtc3lzdGVtLCBcIkhlbHZldGljYSBOZXVlXCIsIEFyaWFsLCBzYW5zLXNlcmlmJyxcclxuICAgIG1vbm86ICdNZW5sbywgTW9uYWNvLCBtb25vc3BhY2UnLFxyXG4gIH0sXHJcbiAgc3BhY2U6IHtcclxuICAgIHBhZGRpbmc6ICc0cHgnLFxyXG4gICAgc3BhY2VCZXR3ZWVuOiAnNHB4JyxcclxuICB9LFxyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
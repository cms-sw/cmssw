webpackHotUpdate_N_E("pages/index",{

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vc3R5bGVzL3RoZW1lLnRzIl0sIm5hbWVzIjpbInRoZW1lIiwiY29sb3JzIiwicHJpbWFyeSIsIm1haW4iLCJsaWdodCIsInNlY29uZGFyeSIsImRhcmsiLCJkYXJrQ29udHJhc3QiLCJ0aGlyZHkiLCJjb21tb24iLCJ3aGl0ZSIsImJsYWNrIiwibGlnaHRHcmV5IiwiYmx1ZUdyZXkiLCJub3RpZmljYXRpb24iLCJlcnJvciIsInN1Y2Nlc3MiLCJkYXJrU3VjY2VzcyIsIndhcm5pbmciLCJmb250RmFtaWx5Iiwic2Fuc1NlcmlmIiwibW9ubyIsInNwYWNlIiwicGFkZGluZyIsInNwYWNlQmV0d2VlbiJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBTyxJQUFNQSxLQUFLLEdBQUc7QUFDbkJDLFFBQU0sRUFBRTtBQUNOQyxXQUFPLEVBQUU7QUFDUEMsVUFBSSxFQUFFLE9BREM7QUFFUEMsV0FBSyxFQUFFO0FBRkEsS0FESDtBQUtOQyxhQUFTLEVBQUU7QUFDVEYsVUFBSSxFQUFFLFNBREc7QUFFVEMsV0FBSyxFQUFFLFNBRkU7QUFHVEUsVUFBSSxFQUFFLFNBSEc7QUFJVEMsa0JBQVksRUFBRTtBQUpMLEtBTEw7QUFXTkMsVUFBTSxFQUFFO0FBQ05GLFVBQUksRUFBRSxTQURBO0FBRU5GLFdBQUssRUFBRTtBQUZELEtBWEY7QUFlTkssVUFBTSxFQUFFO0FBQ05DLFdBQUssRUFBRSxNQUREO0FBRU5DLFdBQUssRUFBRSxNQUZEO0FBR05DLGVBQVMsRUFBRSxTQUhMO0FBSU5DLGNBQVEsRUFBRTtBQUpKLEtBZkY7QUFxQk5DLGdCQUFZLEVBQUU7QUFDWkMsV0FBSyxFQUFFLFNBREs7QUFFWkMsYUFBTyxFQUFFLFNBRkc7QUFHWkMsaUJBQVcsRUFBRSxTQUhEO0FBSVpDLGFBQU8sRUFBRTtBQUpHO0FBckJSLEdBRFc7QUE2Qm5CQyxZQUFVLEVBQUU7QUFDVkMsYUFBUyxFQUFFLG9EQUREO0FBRVZDLFFBQUksRUFBRTtBQUZJLEdBN0JPO0FBaUNuQkMsT0FBSyxFQUFFO0FBQ0xDLFdBQU8sRUFBRSxLQURKO0FBRUxDLGdCQUFZLEVBQUU7QUFGVDtBQWpDWSxDQUFkIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjkyN2ZjMjVjZmFhYWNlODRiMjIzLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJleHBvcnQgY29uc3QgdGhlbWUgPSB7XHJcbiAgY29sb3JzOiB7XHJcbiAgICBwcmltYXJ5OiB7XHJcbiAgICAgIG1haW46ICdncmVlbicsXHJcbiAgICAgIGxpZ2h0OiAnI0M1Q0JFMycsXHJcbiAgICB9LFxyXG4gICAgc2Vjb25kYXJ5OiB7XHJcbiAgICAgIG1haW46ICcjQUMzQjYxJyxcclxuICAgICAgbGlnaHQ6ICcjZWRjN2I3JyxcclxuICAgICAgZGFyazogJyM1ZDFkMzInLFxyXG4gICAgICBkYXJrQ29udHJhc3Q6ICcjZmZmJyxcclxuICAgIH0sXHJcbiAgICB0aGlyZHk6IHtcclxuICAgICAgZGFyazogJyM0QzZCOUQnLFxyXG4gICAgICBsaWdodDogJyNBNEFCQkQnLFxyXG4gICAgfSxcclxuICAgIGNvbW1vbjoge1xyXG4gICAgICB3aGl0ZTogJyNmZmYnLFxyXG4gICAgICBibGFjazogJyMwMDAnLFxyXG4gICAgICBsaWdodEdyZXk6ICcjZGZkZmRmJyxcclxuICAgICAgYmx1ZUdyZXk6ICcjZjBmMmY1JyxcclxuICAgIH0sXHJcbiAgICBub3RpZmljYXRpb246IHtcclxuICAgICAgZXJyb3I6ICcjZDAxZjFmJyxcclxuICAgICAgc3VjY2VzczogJyM0MWMxMmQnLFxyXG4gICAgICBkYXJrU3VjY2VzczogJyMzZWIwMmMnLFxyXG4gICAgICB3YXJuaW5nOiAnI2YwOGUxMycsXHJcbiAgICB9LFxyXG4gIH0sXHJcbiAgZm9udEZhbWlseToge1xyXG4gICAgc2Fuc1NlcmlmOiAnLWFwcGxlLXN5c3RlbSwgXCJIZWx2ZXRpY2EgTmV1ZVwiLCBBcmlhbCwgc2Fucy1zZXJpZicsXHJcbiAgICBtb25vOiAnTWVubG8sIE1vbmFjbywgbW9ub3NwYWNlJyxcclxuICB9LFxyXG4gIHNwYWNlOiB7XHJcbiAgICBwYWRkaW5nOiAnNHB4JyxcclxuICAgIHNwYWNlQmV0d2VlbjogJzRweCcsXHJcbiAgfSxcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
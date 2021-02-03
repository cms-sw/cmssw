webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotsWithLayouts/styledComponents.ts":
/*!********************************************************************!*\
  !*** ./components/plots/plot/plotsWithLayouts/styledComponents.ts ***!
  \********************************************************************/
/*! exports provided: ParentWrapper, LayoutName, LayoutWrapper, PlotWrapper */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ParentWrapper", function() { return ParentWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LayoutName", function() { return LayoutName; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LayoutWrapper", function() { return LayoutWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotWrapper", function() { return PlotWrapper; });
/* harmony import */ var styled_components__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! styled-components */ "./node_modules/styled-components/dist/styled-components.browser.esm.js");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../../../styles/theme */ "./styles/theme.ts");


var keyframe_for_updates_plots = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["keyframes"])(["0%{background:", ";color:", ";}100%{background:", ";}"], _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.main, _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.common.white, _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
var ParentWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__ParentWrapper",
  componentId: "qjilkp-0"
})(["width:", "px;height:", "px;justify-content:center;margin:4px;background:", ";display:grid;align-items:end;padding:8px;animation-iteration-count:1;animation-duration:1s;animation-name:", ";"], function (props) {
  return props.size.w + 30 + (props.plotsAmount ? props.plotsAmount : 4 * 4);
}, function (props) {
  return props.size.h + 40 + (props.plotsAmount ? props.plotsAmount : 4 * 4);
}, function (props) {
  return props.isPlotSelected === 'true' ? _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light : _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light;
}, function (props) {
  return props.isLoading === 'true' && props.animation === 'true' ? keyframe_for_updates_plots : '';
});
var LayoutName = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__LayoutName",
  componentId: "qjilkp-1"
})(["padding-bottom:4;color:", ";font-weight:", ";word-break:break-word;"], function (props) {
  return props.error === 'true' ? _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.notification.error : _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.common.black;
}, function (props) {
  return props.isPlotSelected === 'true' ? 'bold' : '';
});
var LayoutWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__LayoutWrapper",
  componentId: "qjilkp-2"
})(["display:grid;grid-template-columns:", ";justify-content:center;"], function (props) {
  return props.auto;
});
var PlotWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__PlotWrapper",
  componentId: "qjilkp-3"
})(["justify-content:center;border:", ";align-items:center;width:", ";height:", ";;cursor:pointer;padding:4px;align-self:center;justify-self:baseline;cursor:", ";"], function (props) {
  return props.plotSelected ? "4px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light) : "2px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
}, function (props) {
  return props.width ? "calc(".concat(props.width, "+8px)") : 'fit-content';
}, function (props) {
  return props.height ? "calc(".concat(props.height, "+8px)") : 'fit-content';
}, function (props) {
  return props.plotSelected ? 'zoom-out' : 'zoom-in';
});

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvc3R5bGVkQ29tcG9uZW50cy50cyJdLCJuYW1lcyI6WyJrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyIsImtleWZyYW1lcyIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsImNvbW1vbiIsIndoaXRlIiwicHJpbWFyeSIsImxpZ2h0IiwiUGFyZW50V3JhcHBlciIsInN0eWxlZCIsImRpdiIsInByb3BzIiwic2l6ZSIsInciLCJwbG90c0Ftb3VudCIsImgiLCJpc1Bsb3RTZWxlY3RlZCIsImlzTG9hZGluZyIsImFuaW1hdGlvbiIsIkxheW91dE5hbWUiLCJlcnJvciIsIm5vdGlmaWNhdGlvbiIsImJsYWNrIiwiTGF5b3V0V3JhcHBlciIsImF1dG8iLCJQbG90V3JhcHBlciIsInBsb3RTZWxlY3RlZCIsIndpZHRoIiwiaGVpZ2h0Il0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUVBO0FBRUEsSUFBTUEsMEJBQTBCLEdBQUdDLG1FQUFILDREQUVkQyxtREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRlQsRUFHbEJILG1EQUFLLENBQUNDLE1BQU4sQ0FBYUcsTUFBYixDQUFvQkMsS0FIRixFQU1kTCxtREFBSyxDQUFDQyxNQUFOLENBQWFLLE9BQWIsQ0FBcUJDLEtBTlAsQ0FBaEM7QUFXTyxJQUFNQyxhQUFhLEdBQUdDLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEscU1BQ2IsVUFBQ0MsS0FBRDtBQUFBLFNBQVlBLEtBQUssQ0FBQ0MsSUFBTixDQUFXQyxDQUFYLEdBQWUsRUFBZixJQUFxQkYsS0FBSyxDQUFDRyxXQUFOLEdBQW9CSCxLQUFLLENBQUNHLFdBQTFCLEdBQXdDLElBQUksQ0FBakUsQ0FBWjtBQUFBLENBRGEsRUFFWixVQUFDSCxLQUFEO0FBQUEsU0FBWUEsS0FBSyxDQUFDQyxJQUFOLENBQVdHLENBQVgsR0FBZSxFQUFmLElBQXFCSixLQUFLLENBQUNHLFdBQU4sR0FBb0JILEtBQUssQ0FBQ0csV0FBMUIsR0FBd0MsSUFBSSxDQUFqRSxDQUFaO0FBQUEsQ0FGWSxFQUtSLFVBQUNILEtBQUQ7QUFBQSxTQUFXQSxLQUFLLENBQUNLLGNBQU4sS0FBeUIsTUFBekIsR0FBa0NoQixtREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJLLEtBQXpELEdBQWlFUCxtREFBSyxDQUFDQyxNQUFOLENBQWFLLE9BQWIsQ0FBcUJDLEtBQWpHO0FBQUEsQ0FMUSxFQVdKLFVBQUNJLEtBQUQ7QUFBQSxTQUNsQkEsS0FBSyxDQUFDTSxTQUFOLEtBQW9CLE1BQXBCLElBQThCTixLQUFLLENBQUNPLFNBQU4sS0FBb0IsTUFBbEQsR0FDSXBCLDBCQURKLEdBRUksRUFIYztBQUFBLENBWEksQ0FBbkI7QUFpQkEsSUFBTXFCLFVBQVUsR0FBR1YseURBQU0sQ0FBQ0MsR0FBVjtBQUFBO0FBQUE7QUFBQSw0RUFFVixVQUFBQyxLQUFLO0FBQUEsU0FBSUEsS0FBSyxDQUFDUyxLQUFOLEtBQWdCLE1BQWhCLEdBQXlCcEIsbURBQUssQ0FBQ0MsTUFBTixDQUFhb0IsWUFBYixDQUEwQkQsS0FBbkQsR0FBMkRwQixtREFBSyxDQUFDQyxNQUFOLENBQWFHLE1BQWIsQ0FBb0JrQixLQUFuRjtBQUFBLENBRkssRUFHSixVQUFBWCxLQUFLO0FBQUEsU0FBSUEsS0FBSyxDQUFDSyxjQUFOLEtBQXlCLE1BQXpCLEdBQWtDLE1BQWxDLEdBQTJDLEVBQS9DO0FBQUEsQ0FIRCxDQUFoQjtBQU1BLElBQU1PLGFBQWEsR0FBR2QseURBQU0sQ0FBQ0MsR0FBVjtBQUFBO0FBQUE7QUFBQSx3RUFJRyxVQUFDQyxLQUFEO0FBQUEsU0FBWUEsS0FBSyxDQUFDYSxJQUFsQjtBQUFBLENBSkgsQ0FBbkI7QUFRQSxJQUFNQyxXQUFXLEdBQUdoQix5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLHNLQUVWLFVBQUNDLEtBQUQ7QUFBQSxTQUFXQSxLQUFLLENBQUNlLFlBQU4sdUJBQWtDMUIsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxTQUFiLENBQXVCSyxLQUF6RCx3QkFBZ0ZQLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUssT0FBYixDQUFxQkMsS0FBckcsQ0FBWDtBQUFBLENBRlUsRUFJWCxVQUFDSSxLQUFEO0FBQUEsU0FBV0EsS0FBSyxDQUFDZ0IsS0FBTixrQkFBc0JoQixLQUFLLENBQUNnQixLQUE1QixhQUEyQyxhQUF0RDtBQUFBLENBSlcsRUFLVixVQUFDaEIsS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ2lCLE1BQU4sa0JBQXVCakIsS0FBSyxDQUFDaUIsTUFBN0IsYUFBNkMsYUFBeEQ7QUFBQSxDQUxVLEVBVVYsVUFBQWpCLEtBQUs7QUFBQSxTQUFJQSxLQUFLLENBQUNlLFlBQU4sR0FBcUIsVUFBckIsR0FBa0MsU0FBdEM7QUFBQSxDQVZLLENBQWpCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4Ljk2ZWRjNmQxZTlmNjI3MzdmMGFhLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgc3R5bGVkLCB7IGtleWZyYW1lcyB9IGZyb20gJ3N0eWxlZC1jb21wb25lbnRzJztcclxuaW1wb3J0IHsgU2l6ZVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uLy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcblxyXG5jb25zdCBrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyA9IGtleWZyYW1lc2BcclxuICAwJSB7XHJcbiAgICBiYWNrZ3JvdW5kOiAke3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn07XHJcbiAgICBjb2xvcjogICR7dGhlbWUuY29sb3JzLmNvbW1vbi53aGl0ZX07XHJcbiAgfVxyXG4gIDEwMCUge1xyXG4gICAgYmFja2dyb3VuZDogJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH07XHJcbiAgfVxyXG5gO1xyXG5cclxuXHJcbmV4cG9ydCBjb25zdCBQYXJlbnRXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHNpemU6IFNpemVQcm9wcywgaXNMb2FkaW5nOiBzdHJpbmcsIGFuaW1hdGlvbjogc3RyaW5nLCBpc1Bsb3RTZWxlY3RlZD86IHN0cmluZywgcGxvdHNBbW91bnQ/OiBudW1iZXI7IH0+YFxyXG4gICAgd2lkdGg6ICR7KHByb3BzKSA9PiAocHJvcHMuc2l6ZS53ICsgMzAgKyAocHJvcHMucGxvdHNBbW91bnQgPyBwcm9wcy5wbG90c0Ftb3VudCA6IDQgKiA0KSl9cHg7XHJcbiAgICBoZWlnaHQ6ICR7KHByb3BzKSA9PiAocHJvcHMuc2l6ZS5oICsgNDAgKyAocHJvcHMucGxvdHNBbW91bnQgPyBwcm9wcy5wbG90c0Ftb3VudCA6IDQgKiA0KSl9cHg7XHJcbiAgICBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjtcclxuICAgIG1hcmdpbjogNHB4O1xyXG4gICAgYmFja2dyb3VuZDogJHsocHJvcHMpID0+IHByb3BzLmlzUGxvdFNlbGVjdGVkID09PSAndHJ1ZScgPyB0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5LmxpZ2h0IDogdGhlbWUuY29sb3JzLnByaW1hcnkubGlnaHR9O1xyXG4gICAgZGlzcGxheTogZ3JpZDtcclxuICAgIGFsaWduLWl0ZW1zOiBlbmQ7XHJcbiAgICBwYWRkaW5nOiA4cHg7XHJcbiAgICBhbmltYXRpb24taXRlcmF0aW9uLWNvdW50OiAxO1xyXG4gICAgYW5pbWF0aW9uLWR1cmF0aW9uOiAxcztcclxuICAgIGFuaW1hdGlvbi1uYW1lOiAkeyhwcm9wcykgPT5cclxuICAgIHByb3BzLmlzTG9hZGluZyA9PT0gJ3RydWUnICYmIHByb3BzLmFuaW1hdGlvbiA9PT0gJ3RydWUnXHJcbiAgICAgID8ga2V5ZnJhbWVfZm9yX3VwZGF0ZXNfcGxvdHNcclxuICAgICAgOiAnJ307XHJcbmBcclxuXHJcbmV4cG9ydCBjb25zdCBMYXlvdXROYW1lID0gc3R5bGVkLmRpdjx7IGVycm9yPzogc3RyaW5nLCBpc1Bsb3RTZWxlY3RlZD86IHN0cmluZyB9PmBcclxuICAgIHBhZGRpbmctYm90dG9tOiA0O1xyXG4gICAgY29sb3I6ICR7cHJvcHMgPT4gcHJvcHMuZXJyb3IgPT09ICd0cnVlJyA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uZXJyb3IgOiB0aGVtZS5jb2xvcnMuY29tbW9uLmJsYWNrfTtcclxuICAgIGZvbnQtd2VpZ2h0OiAke3Byb3BzID0+IHByb3BzLmlzUGxvdFNlbGVjdGVkID09PSAndHJ1ZScgPyAnYm9sZCcgOiAnJ307XHJcbiAgICB3b3JkLWJyZWFrOiBicmVhay13b3JkO1xyXG5gXHJcbmV4cG9ydCBjb25zdCBMYXlvdXRXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHNpemU6IFNpemVQcm9wcyAmIHN0cmluZywgYXV0bzogc3RyaW5nIH0+YFxyXG4gICAgLy8gd2lkdGg6ICR7KHByb3BzKSA9PiBwcm9wcy5zaXplLncgPyBgJHtwcm9wcy5zaXplLncgKyAxMn1weGAgOiBwcm9wcy5zaXplfTtcclxuICAgIC8vIGhlaWdodDokeyhwcm9wcykgPT4gcHJvcHMuc2l6ZS5oID8gYCR7cHJvcHMuc2l6ZS53ICsgMTZ9cHhgIDogcHJvcHMuc2l6ZX07XHJcbiAgICBkaXNwbGF5OiBncmlkO1xyXG4gICAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAkeyhwcm9wcykgPT4gKHByb3BzLmF1dG8pfTtcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG5gO1xyXG5cclxuZXhwb3J0IGNvbnN0IFBsb3RXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHBsb3RTZWxlY3RlZDogYm9vbGVhbiwgd2lkdGg/OiBzdHJpbmcsIGhlaWdodD86IHN0cmluZyB9PmBcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG4gICAgYm9yZGVyOiAkeyhwcm9wcykgPT4gcHJvcHMucGxvdFNlbGVjdGVkID8gYDRweCBzb2xpZCAke3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubGlnaHR9YCA6IGAycHggc29saWQgJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH1gfTtcclxuICAgIGFsaWduLWl0ZW1zOiAgY2VudGVyIDtcclxuICAgIHdpZHRoOiAkeyhwcm9wcykgPT4gcHJvcHMud2lkdGggPyBgY2FsYygke3Byb3BzLndpZHRofSs4cHgpYCA6ICdmaXQtY29udGVudCd9O1xyXG4gICAgaGVpZ2h0OiAkeyhwcm9wcykgPT4gcHJvcHMuaGVpZ2h0ID8gYGNhbGMoJHtwcm9wcy5oZWlnaHR9KzhweClgIDogJ2ZpdC1jb250ZW50J307O1xyXG4gICAgY3Vyc29yOiAgcG9pbnRlciA7XHJcbiAgICBwYWRkaW5nOiA0cHg7XHJcbiAgICBhbGlnbi1zZWxmOiAgY2VudGVyIDtcclxuICAgIGp1c3RpZnktc2VsZjogIGJhc2VsaW5lO1xyXG4gICAgY3Vyc29yOiAke3Byb3BzID0+IHByb3BzLnBsb3RTZWxlY3RlZCA/ICd6b29tLW91dCcgOiAnem9vbS1pbid9O1xyXG5gIl0sInNvdXJjZVJvb3QiOiIifQ==
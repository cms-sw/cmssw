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
})(["width:", "px;justify-content:center;margin:4px;background:", ";display:grid;align-items:end;padding:8px;animation-iteration-count:1;animation-duration:1s;animation-name:", ";"], function (props) {
  return props.size.w + 24 + (props.plotsAmount ? props.plotsAmount : 0 * 4);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvc3R5bGVkQ29tcG9uZW50cy50cyJdLCJuYW1lcyI6WyJrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyIsImtleWZyYW1lcyIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsImNvbW1vbiIsIndoaXRlIiwicHJpbWFyeSIsImxpZ2h0IiwiUGFyZW50V3JhcHBlciIsInN0eWxlZCIsImRpdiIsInByb3BzIiwic2l6ZSIsInciLCJwbG90c0Ftb3VudCIsImlzUGxvdFNlbGVjdGVkIiwiaXNMb2FkaW5nIiwiYW5pbWF0aW9uIiwiTGF5b3V0TmFtZSIsImVycm9yIiwibm90aWZpY2F0aW9uIiwiYmxhY2siLCJMYXlvdXRXcmFwcGVyIiwiYXV0byIsIlBsb3RXcmFwcGVyIiwicGxvdFNlbGVjdGVkIiwid2lkdGgiLCJoZWlnaHQiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBRUE7QUFFQSxJQUFNQSwwQkFBMEIsR0FBR0MsbUVBQUgsNERBRWRDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsU0FBYixDQUF1QkMsSUFGVCxFQUdsQkgsbURBQUssQ0FBQ0MsTUFBTixDQUFhRyxNQUFiLENBQW9CQyxLQUhGLEVBTWRMLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUssT0FBYixDQUFxQkMsS0FOUCxDQUFoQztBQVdPLElBQU1DLGFBQWEsR0FBR0MseURBQU0sQ0FBQ0MsR0FBVjtBQUFBO0FBQUE7QUFBQSx1TEFDYixVQUFDQyxLQUFEO0FBQUEsU0FBWUEsS0FBSyxDQUFDQyxJQUFOLENBQVdDLENBQVgsR0FBZSxFQUFmLElBQXFCRixLQUFLLENBQUNHLFdBQU4sR0FBb0JILEtBQUssQ0FBQ0csV0FBMUIsR0FBd0MsSUFBSSxDQUFqRSxDQUFaO0FBQUEsQ0FEYSxFQUlSLFVBQUNILEtBQUQ7QUFBQSxTQUFXQSxLQUFLLENBQUNJLGNBQU4sS0FBeUIsTUFBekIsR0FBa0NmLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsU0FBYixDQUF1QkssS0FBekQsR0FBaUVQLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUssT0FBYixDQUFxQkMsS0FBakc7QUFBQSxDQUpRLEVBVUosVUFBQ0ksS0FBRDtBQUFBLFNBQ2xCQSxLQUFLLENBQUNLLFNBQU4sS0FBb0IsTUFBcEIsSUFBOEJMLEtBQUssQ0FBQ00sU0FBTixLQUFvQixNQUFsRCxHQUNJbkIsMEJBREosR0FFSSxFQUhjO0FBQUEsQ0FWSSxDQUFuQjtBQWdCQSxJQUFNb0IsVUFBVSxHQUFHVCx5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLDRFQUVWLFVBQUFDLEtBQUs7QUFBQSxTQUFJQSxLQUFLLENBQUNRLEtBQU4sS0FBZ0IsTUFBaEIsR0FBeUJuQixtREFBSyxDQUFDQyxNQUFOLENBQWFtQixZQUFiLENBQTBCRCxLQUFuRCxHQUEyRG5CLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUcsTUFBYixDQUFvQmlCLEtBQW5GO0FBQUEsQ0FGSyxFQUdKLFVBQUFWLEtBQUs7QUFBQSxTQUFJQSxLQUFLLENBQUNJLGNBQU4sS0FBeUIsTUFBekIsR0FBa0MsTUFBbEMsR0FBMkMsRUFBL0M7QUFBQSxDQUhELENBQWhCO0FBTUEsSUFBTU8sYUFBYSxHQUFHYix5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLHdFQUlHLFVBQUNDLEtBQUQ7QUFBQSxTQUFZQSxLQUFLLENBQUNZLElBQWxCO0FBQUEsQ0FKSCxDQUFuQjtBQVFBLElBQU1DLFdBQVcsR0FBR2YseURBQU0sQ0FBQ0MsR0FBVjtBQUFBO0FBQUE7QUFBQSxzS0FFVixVQUFDQyxLQUFEO0FBQUEsU0FBV0EsS0FBSyxDQUFDYyxZQUFOLHVCQUFrQ3pCLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsU0FBYixDQUF1QkssS0FBekQsd0JBQWdGUCxtREFBSyxDQUFDQyxNQUFOLENBQWFLLE9BQWIsQ0FBcUJDLEtBQXJHLENBQVg7QUFBQSxDQUZVLEVBSVgsVUFBQ0ksS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ2UsS0FBTixrQkFBc0JmLEtBQUssQ0FBQ2UsS0FBNUIsYUFBMkMsYUFBdEQ7QUFBQSxDQUpXLEVBS1YsVUFBQ2YsS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ2dCLE1BQU4sa0JBQXVCaEIsS0FBSyxDQUFDZ0IsTUFBN0IsYUFBNkMsYUFBeEQ7QUFBQSxDQUxVLEVBVVYsVUFBQWhCLEtBQUs7QUFBQSxTQUFJQSxLQUFLLENBQUNjLFlBQU4sR0FBcUIsVUFBckIsR0FBa0MsU0FBdEM7QUFBQSxDQVZLLENBQWpCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmYyNjNmZmRkNmFkNDk5MTBlZDFlLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgc3R5bGVkLCB7IGtleWZyYW1lcyB9IGZyb20gJ3N0eWxlZC1jb21wb25lbnRzJztcclxuaW1wb3J0IHsgU2l6ZVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uLy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcblxyXG5jb25zdCBrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyA9IGtleWZyYW1lc2BcclxuICAwJSB7XHJcbiAgICBiYWNrZ3JvdW5kOiAke3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn07XHJcbiAgICBjb2xvcjogICR7dGhlbWUuY29sb3JzLmNvbW1vbi53aGl0ZX07XHJcbiAgfVxyXG4gIDEwMCUge1xyXG4gICAgYmFja2dyb3VuZDogJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH07XHJcbiAgfVxyXG5gO1xyXG5cclxuXHJcbmV4cG9ydCBjb25zdCBQYXJlbnRXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHNpemU6IFNpemVQcm9wcywgaXNMb2FkaW5nOiBzdHJpbmcsIGFuaW1hdGlvbjogc3RyaW5nLCBpc1Bsb3RTZWxlY3RlZD86IHN0cmluZywgcGxvdHNBbW91bnQ/OiBudW1iZXI7IH0+YFxyXG4gICAgd2lkdGg6ICR7KHByb3BzKSA9PiAocHJvcHMuc2l6ZS53ICsgMjQgKyAocHJvcHMucGxvdHNBbW91bnQgPyBwcm9wcy5wbG90c0Ftb3VudCA6IDAgKiA0KSl9cHg7XHJcbiAgICBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjtcclxuICAgIG1hcmdpbjogNHB4O1xyXG4gICAgYmFja2dyb3VuZDogJHsocHJvcHMpID0+IHByb3BzLmlzUGxvdFNlbGVjdGVkID09PSAndHJ1ZScgPyB0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5LmxpZ2h0IDogdGhlbWUuY29sb3JzLnByaW1hcnkubGlnaHR9O1xyXG4gICAgZGlzcGxheTogZ3JpZDtcclxuICAgIGFsaWduLWl0ZW1zOiBlbmQ7XHJcbiAgICBwYWRkaW5nOiA4cHg7XHJcbiAgICBhbmltYXRpb24taXRlcmF0aW9uLWNvdW50OiAxO1xyXG4gICAgYW5pbWF0aW9uLWR1cmF0aW9uOiAxcztcclxuICAgIGFuaW1hdGlvbi1uYW1lOiAkeyhwcm9wcykgPT5cclxuICAgIHByb3BzLmlzTG9hZGluZyA9PT0gJ3RydWUnICYmIHByb3BzLmFuaW1hdGlvbiA9PT0gJ3RydWUnXHJcbiAgICAgID8ga2V5ZnJhbWVfZm9yX3VwZGF0ZXNfcGxvdHNcclxuICAgICAgOiAnJ307XHJcbmBcclxuXHJcbmV4cG9ydCBjb25zdCBMYXlvdXROYW1lID0gc3R5bGVkLmRpdjx7IGVycm9yPzogc3RyaW5nLCBpc1Bsb3RTZWxlY3RlZD86IHN0cmluZyB9PmBcclxuICAgIHBhZGRpbmctYm90dG9tOiA0O1xyXG4gICAgY29sb3I6ICR7cHJvcHMgPT4gcHJvcHMuZXJyb3IgPT09ICd0cnVlJyA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uZXJyb3IgOiB0aGVtZS5jb2xvcnMuY29tbW9uLmJsYWNrfTtcclxuICAgIGZvbnQtd2VpZ2h0OiAke3Byb3BzID0+IHByb3BzLmlzUGxvdFNlbGVjdGVkID09PSAndHJ1ZScgPyAnYm9sZCcgOiAnJ307XHJcbiAgICB3b3JkLWJyZWFrOiBicmVhay13b3JkO1xyXG5gXHJcbmV4cG9ydCBjb25zdCBMYXlvdXRXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHNpemU6IFNpemVQcm9wcyAmIHN0cmluZywgYXV0bzogc3RyaW5nIH0+YFxyXG4gICAgLy8gd2lkdGg6ICR7KHByb3BzKSA9PiBwcm9wcy5zaXplLncgPyBgJHtwcm9wcy5zaXplLncgKyAxMn1weGAgOiBwcm9wcy5zaXplfTtcclxuICAgIC8vIGhlaWdodDokeyhwcm9wcykgPT4gcHJvcHMuc2l6ZS5oID8gYCR7cHJvcHMuc2l6ZS53ICsgMTZ9cHhgIDogcHJvcHMuc2l6ZX07XHJcbiAgICBkaXNwbGF5OiBncmlkO1xyXG4gICAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAkeyhwcm9wcykgPT4gKHByb3BzLmF1dG8pfTtcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG5gO1xyXG5cclxuZXhwb3J0IGNvbnN0IFBsb3RXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHBsb3RTZWxlY3RlZDogYm9vbGVhbiwgd2lkdGg6IHN0cmluZywgaGVpZ2h0OiBzdHJpbmcgfT5gXHJcbiAgICBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjtcclxuICAgIGJvcmRlcjogJHsocHJvcHMpID0+IHByb3BzLnBsb3RTZWxlY3RlZCA/IGA0cHggc29saWQgJHt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5LmxpZ2h0fWAgOiBgMnB4IHNvbGlkICR7dGhlbWUuY29sb3JzLnByaW1hcnkubGlnaHR9YH07XHJcbiAgICBhbGlnbi1pdGVtczogIGNlbnRlciA7XHJcbiAgICB3aWR0aDogJHsocHJvcHMpID0+IHByb3BzLndpZHRoID8gYGNhbGMoJHtwcm9wcy53aWR0aH0rOHB4KWAgOiAnZml0LWNvbnRlbnQnfTtcclxuICAgIGhlaWdodDogJHsocHJvcHMpID0+IHByb3BzLmhlaWdodCA/IGBjYWxjKCR7cHJvcHMuaGVpZ2h0fSs4cHgpYCA6ICdmaXQtY29udGVudCd9OztcclxuICAgIGN1cnNvcjogIHBvaW50ZXIgO1xyXG4gICAgcGFkZGluZzogNHB4O1xyXG4gICAgYWxpZ24tc2VsZjogIGNlbnRlciA7XHJcbiAgICBqdXN0aWZ5LXNlbGY6ICBiYXNlbGluZTtcclxuICAgIGN1cnNvcjogJHtwcm9wcyA9PiBwcm9wcy5wbG90U2VsZWN0ZWQgPyAnem9vbS1vdXQnIDogJ3pvb20taW4nfTtcclxuYCJdLCJzb3VyY2VSb290IjoiIn0=
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
  return props.size.w + 24;
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
})(["justify-content:center;border:", ";align-items:center;width:", ";height:", ";;cursor:pointer;align-self:center;justify-self:baseline;margin:2px;cursor:", ";"], function (props) {
  return props.plotSelected ? "4px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light) : "2px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
}, function (props) {
  return props.width ? props.width : 'fit-content';
}, function (props) {
  return props.height ? props.height : 'fit-content';
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvc3R5bGVkQ29tcG9uZW50cy50cyJdLCJuYW1lcyI6WyJrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyIsImtleWZyYW1lcyIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsImNvbW1vbiIsIndoaXRlIiwicHJpbWFyeSIsImxpZ2h0IiwiUGFyZW50V3JhcHBlciIsInN0eWxlZCIsImRpdiIsInByb3BzIiwic2l6ZSIsInciLCJpc1Bsb3RTZWxlY3RlZCIsImlzTG9hZGluZyIsImFuaW1hdGlvbiIsIkxheW91dE5hbWUiLCJlcnJvciIsIm5vdGlmaWNhdGlvbiIsImJsYWNrIiwiTGF5b3V0V3JhcHBlciIsImF1dG8iLCJQbG90V3JhcHBlciIsInBsb3RTZWxlY3RlZCIsIndpZHRoIiwiaGVpZ2h0Il0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUVBO0FBRUEsSUFBTUEsMEJBQTBCLEdBQUdDLG1FQUFILDREQUVkQyxtREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRlQsRUFHbEJILG1EQUFLLENBQUNDLE1BQU4sQ0FBYUcsTUFBYixDQUFvQkMsS0FIRixFQU1kTCxtREFBSyxDQUFDQyxNQUFOLENBQWFLLE9BQWIsQ0FBcUJDLEtBTlAsQ0FBaEM7QUFXTyxJQUFNQyxhQUFhLEdBQUdDLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsdUxBQ2IsVUFBQ0MsS0FBRDtBQUFBLFNBQVlBLEtBQUssQ0FBQ0MsSUFBTixDQUFXQyxDQUFYLEdBQWUsRUFBM0I7QUFBQSxDQURhLEVBSVIsVUFBQ0YsS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ0csY0FBTixLQUF5QixNQUF6QixHQUFrQ2QsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxTQUFiLENBQXVCSyxLQUF6RCxHQUFpRVAsbURBQUssQ0FBQ0MsTUFBTixDQUFhSyxPQUFiLENBQXFCQyxLQUFqRztBQUFBLENBSlEsRUFVSixVQUFDSSxLQUFEO0FBQUEsU0FDbEJBLEtBQUssQ0FBQ0ksU0FBTixLQUFvQixNQUFwQixJQUE4QkosS0FBSyxDQUFDSyxTQUFOLEtBQW9CLE1BQWxELEdBQ0lsQiwwQkFESixHQUVJLEVBSGM7QUFBQSxDQVZJLENBQW5CO0FBZ0JBLElBQU1tQixVQUFVLEdBQUdSLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsNEVBRVYsVUFBQUMsS0FBSztBQUFBLFNBQUlBLEtBQUssQ0FBQ08sS0FBTixLQUFnQixNQUFoQixHQUF5QmxCLG1EQUFLLENBQUNDLE1BQU4sQ0FBYWtCLFlBQWIsQ0FBMEJELEtBQW5ELEdBQTJEbEIsbURBQUssQ0FBQ0MsTUFBTixDQUFhRyxNQUFiLENBQW9CZ0IsS0FBbkY7QUFBQSxDQUZLLEVBR0osVUFBQVQsS0FBSztBQUFBLFNBQUlBLEtBQUssQ0FBQ0csY0FBTixLQUF5QixNQUF6QixHQUFrQyxNQUFsQyxHQUEyQyxFQUEvQztBQUFBLENBSEQsQ0FBaEI7QUFNQSxJQUFNTyxhQUFhLEdBQUdaLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsd0VBSUcsVUFBQ0MsS0FBRDtBQUFBLFNBQVlBLEtBQUssQ0FBQ1csSUFBbEI7QUFBQSxDQUpILENBQW5CO0FBUUEsSUFBTUMsV0FBVyxHQUFHZCx5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLHFLQUVWLFVBQUNDLEtBQUQ7QUFBQSxTQUFXQSxLQUFLLENBQUNhLFlBQU4sdUJBQWtDeEIsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxTQUFiLENBQXVCSyxLQUF6RCx3QkFBZ0ZQLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUssT0FBYixDQUFxQkMsS0FBckcsQ0FBWDtBQUFBLENBRlUsRUFJWCxVQUFDSSxLQUFEO0FBQUEsU0FBV0EsS0FBSyxDQUFDYyxLQUFOLEdBQWNkLEtBQUssQ0FBQ2MsS0FBcEIsR0FBNEIsYUFBdkM7QUFBQSxDQUpXLEVBS1YsVUFBQ2QsS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ2UsTUFBTixHQUFlZixLQUFLLENBQUNlLE1BQXJCLEdBQThCLGFBQXpDO0FBQUEsQ0FMVSxFQVVWLFVBQUFmLEtBQUs7QUFBQSxTQUFJQSxLQUFLLENBQUNhLFlBQU4sR0FBcUIsVUFBckIsR0FBa0MsU0FBdEM7QUFBQSxDQVZLLENBQWpCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjIxMTEzOTE2OWE4NzgzMzkyZjg5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgc3R5bGVkLCB7IGtleWZyYW1lcyB9IGZyb20gJ3N0eWxlZC1jb21wb25lbnRzJztcclxuaW1wb3J0IHsgU2l6ZVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uLy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcblxyXG5jb25zdCBrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyA9IGtleWZyYW1lc2BcclxuICAwJSB7XHJcbiAgICBiYWNrZ3JvdW5kOiAke3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn07XHJcbiAgICBjb2xvcjogICR7dGhlbWUuY29sb3JzLmNvbW1vbi53aGl0ZX07XHJcbiAgfVxyXG4gIDEwMCUge1xyXG4gICAgYmFja2dyb3VuZDogJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH07XHJcbiAgfVxyXG5gO1xyXG5cclxuXHJcbmV4cG9ydCBjb25zdCBQYXJlbnRXcmFwcGVyID0gc3R5bGVkLmRpdjx7IHNpemU6IFNpemVQcm9wcywgaXNMb2FkaW5nOiBzdHJpbmcsIGFuaW1hdGlvbjogc3RyaW5nLCBpc1Bsb3RTZWxlY3RlZD86IHN0cmluZyB9PmBcclxuICAgIHdpZHRoOiAkeyhwcm9wcykgPT4gKHByb3BzLnNpemUudyArIDI0KX1weDtcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG4gICAgbWFyZ2luOiA0cHg7XHJcbiAgICBiYWNrZ3JvdW5kOiAkeyhwcm9wcykgPT4gcHJvcHMuaXNQbG90U2VsZWN0ZWQgPT09ICd0cnVlJyA/IHRoZW1lLmNvbG9ycy5zZWNvbmRhcnkubGlnaHQgOiB0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH07XHJcbiAgICBkaXNwbGF5OiBncmlkO1xyXG4gICAgYWxpZ24taXRlbXM6IGVuZDtcclxuICAgIHBhZGRpbmc6IDhweDtcclxuICAgIGFuaW1hdGlvbi1pdGVyYXRpb24tY291bnQ6IDE7XHJcbiAgICBhbmltYXRpb24tZHVyYXRpb246IDFzO1xyXG4gICAgYW5pbWF0aW9uLW5hbWU6ICR7KHByb3BzKSA9PlxyXG4gICAgcHJvcHMuaXNMb2FkaW5nID09PSAndHJ1ZScgJiYgcHJvcHMuYW5pbWF0aW9uID09PSAndHJ1ZSdcclxuICAgICAgPyBrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90c1xyXG4gICAgICA6ICcnfTtcclxuYFxyXG5cclxuZXhwb3J0IGNvbnN0IExheW91dE5hbWUgPSBzdHlsZWQuZGl2PHsgZXJyb3I/OiBzdHJpbmcsIGlzUGxvdFNlbGVjdGVkPzogc3RyaW5nIH0+YFxyXG4gICAgcGFkZGluZy1ib3R0b206IDQ7XHJcbiAgICBjb2xvcjogJHtwcm9wcyA9PiBwcm9wcy5lcnJvciA9PT0gJ3RydWUnID8gdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5lcnJvciA6IHRoZW1lLmNvbG9ycy5jb21tb24uYmxhY2t9O1xyXG4gICAgZm9udC13ZWlnaHQ6ICR7cHJvcHMgPT4gcHJvcHMuaXNQbG90U2VsZWN0ZWQgPT09ICd0cnVlJyA/ICdib2xkJyA6ICcnfTtcclxuICAgIHdvcmQtYnJlYWs6IGJyZWFrLXdvcmQ7XHJcbmBcclxuZXhwb3J0IGNvbnN0IExheW91dFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgc2l6ZTogU2l6ZVByb3BzICYgc3RyaW5nLCBhdXRvOiBzdHJpbmcgfT5gXHJcbiAgICAvLyB3aWR0aDogJHsocHJvcHMpID0+IHByb3BzLnNpemUudyA/IGAke3Byb3BzLnNpemUudyArIDEyfXB4YCA6IHByb3BzLnNpemV9O1xyXG4gICAgLy8gaGVpZ2h0OiR7KHByb3BzKSA9PiBwcm9wcy5zaXplLmggPyBgJHtwcm9wcy5zaXplLncgKyAxNn1weGAgOiBwcm9wcy5zaXplfTtcclxuICAgIGRpc3BsYXk6IGdyaWQ7XHJcbiAgICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6ICR7KHByb3BzKSA9PiAocHJvcHMuYXV0byl9O1xyXG4gICAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7XHJcbmA7XHJcblxyXG5leHBvcnQgY29uc3QgUGxvdFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgcGxvdFNlbGVjdGVkOiBib29sZWFuLCB3aWR0aDogc3RyaW5nLCBoZWlnaHQ6IHN0cmluZyB9PmBcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG4gICAgYm9yZGVyOiAkeyhwcm9wcykgPT4gcHJvcHMucGxvdFNlbGVjdGVkID8gYDRweCBzb2xpZCAke3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubGlnaHR9YCA6IGAycHggc29saWQgJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH1gfTtcclxuICAgIGFsaWduLWl0ZW1zOiAgY2VudGVyIDtcclxuICAgIHdpZHRoOiAkeyhwcm9wcykgPT4gcHJvcHMud2lkdGggPyBwcm9wcy53aWR0aCA6ICdmaXQtY29udGVudCd9O1xyXG4gICAgaGVpZ2h0OiAkeyhwcm9wcykgPT4gcHJvcHMuaGVpZ2h0ID8gcHJvcHMuaGVpZ2h0IDogJ2ZpdC1jb250ZW50J307O1xyXG4gICAgY3Vyc29yOiAgcG9pbnRlciA7XHJcbiAgICBhbGlnbi1zZWxmOiAgY2VudGVyIDtcclxuICAgIGp1c3RpZnktc2VsZjogIGJhc2VsaW5lO1xyXG4gICAgbWFyZ2luOiAycHg7XHJcbiAgICBjdXJzb3I6ICR7cHJvcHMgPT4gcHJvcHMucGxvdFNlbGVjdGVkID8gJ3pvb20tb3V0JyA6ICd6b29tLWluJ307XHJcbmAiXSwic291cmNlUm9vdCI6IiJ9
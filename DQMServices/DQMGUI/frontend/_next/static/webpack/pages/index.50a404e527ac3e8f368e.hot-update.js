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
})(["justify-content:center;border:", ";align-items:center;width:", ";height:", ";;cursor:pointer;padding:4px;align-self:center;justify-self:baseline;cursor:", ";"], function (props) {
  return props.plotSelected ? "4px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light) : "2px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
}, function (props) {
  return props.width ? "calc(".concat(props.width, "+8px)") : 'fit-content';
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvc3R5bGVkQ29tcG9uZW50cy50cyJdLCJuYW1lcyI6WyJrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyIsImtleWZyYW1lcyIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsImNvbW1vbiIsIndoaXRlIiwicHJpbWFyeSIsImxpZ2h0IiwiUGFyZW50V3JhcHBlciIsInN0eWxlZCIsImRpdiIsInByb3BzIiwic2l6ZSIsInciLCJpc1Bsb3RTZWxlY3RlZCIsImlzTG9hZGluZyIsImFuaW1hdGlvbiIsIkxheW91dE5hbWUiLCJlcnJvciIsIm5vdGlmaWNhdGlvbiIsImJsYWNrIiwiTGF5b3V0V3JhcHBlciIsImF1dG8iLCJQbG90V3JhcHBlciIsInBsb3RTZWxlY3RlZCIsIndpZHRoIiwiaGVpZ2h0Il0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUVBO0FBRUEsSUFBTUEsMEJBQTBCLEdBQUdDLG1FQUFILDREQUVkQyxtREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRlQsRUFHbEJILG1EQUFLLENBQUNDLE1BQU4sQ0FBYUcsTUFBYixDQUFvQkMsS0FIRixFQU1kTCxtREFBSyxDQUFDQyxNQUFOLENBQWFLLE9BQWIsQ0FBcUJDLEtBTlAsQ0FBaEM7QUFXTyxJQUFNQyxhQUFhLEdBQUdDLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsdUxBQ2IsVUFBQ0MsS0FBRDtBQUFBLFNBQVlBLEtBQUssQ0FBQ0MsSUFBTixDQUFXQyxDQUFYLEdBQWUsRUFBM0I7QUFBQSxDQURhLEVBSVIsVUFBQ0YsS0FBRDtBQUFBLFNBQVdBLEtBQUssQ0FBQ0csY0FBTixLQUF5QixNQUF6QixHQUFrQ2QsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxTQUFiLENBQXVCSyxLQUF6RCxHQUFpRVAsbURBQUssQ0FBQ0MsTUFBTixDQUFhSyxPQUFiLENBQXFCQyxLQUFqRztBQUFBLENBSlEsRUFVSixVQUFDSSxLQUFEO0FBQUEsU0FDbEJBLEtBQUssQ0FBQ0ksU0FBTixLQUFvQixNQUFwQixJQUE4QkosS0FBSyxDQUFDSyxTQUFOLEtBQW9CLE1BQWxELEdBQ0lsQiwwQkFESixHQUVJLEVBSGM7QUFBQSxDQVZJLENBQW5CO0FBZ0JBLElBQU1tQixVQUFVLEdBQUdSLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsNEVBRVYsVUFBQUMsS0FBSztBQUFBLFNBQUlBLEtBQUssQ0FBQ08sS0FBTixLQUFnQixNQUFoQixHQUF5QmxCLG1EQUFLLENBQUNDLE1BQU4sQ0FBYWtCLFlBQWIsQ0FBMEJELEtBQW5ELEdBQTJEbEIsbURBQUssQ0FBQ0MsTUFBTixDQUFhRyxNQUFiLENBQW9CZ0IsS0FBbkY7QUFBQSxDQUZLLEVBR0osVUFBQVQsS0FBSztBQUFBLFNBQUlBLEtBQUssQ0FBQ0csY0FBTixLQUF5QixNQUF6QixHQUFrQyxNQUFsQyxHQUEyQyxFQUEvQztBQUFBLENBSEQsQ0FBaEI7QUFNQSxJQUFNTyxhQUFhLEdBQUdaLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsd0VBSUcsVUFBQ0MsS0FBRDtBQUFBLFNBQVlBLEtBQUssQ0FBQ1csSUFBbEI7QUFBQSxDQUpILENBQW5CO0FBUUEsSUFBTUMsV0FBVyxHQUFHZCx5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLHNLQUVWLFVBQUNDLEtBQUQ7QUFBQSxTQUFXQSxLQUFLLENBQUNhLFlBQU4sdUJBQWtDeEIsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxTQUFiLENBQXVCSyxLQUF6RCx3QkFBZ0ZQLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUssT0FBYixDQUFxQkMsS0FBckcsQ0FBWDtBQUFBLENBRlUsRUFJWCxVQUFDSSxLQUFEO0FBQUEsU0FBV0EsS0FBSyxDQUFDYyxLQUFOLGtCQUFzQmQsS0FBSyxDQUFDYyxLQUE1QixhQUEyQyxhQUF0RDtBQUFBLENBSlcsRUFLVixVQUFDZCxLQUFEO0FBQUEsU0FBV0EsS0FBSyxDQUFDZSxNQUFOLEdBQWVmLEtBQUssQ0FBQ2UsTUFBckIsR0FBOEIsYUFBekM7QUFBQSxDQUxVLEVBVVYsVUFBQWYsS0FBSztBQUFBLFNBQUlBLEtBQUssQ0FBQ2EsWUFBTixHQUFxQixVQUFyQixHQUFrQyxTQUF0QztBQUFBLENBVkssQ0FBakIiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNTBhNDA0ZTUyN2FjM2U4ZjM2OGUuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBzdHlsZWQsIHsga2V5ZnJhbWVzIH0gZnJvbSAnc3R5bGVkLWNvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyBTaXplUHJvcHMgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vLi4vLi4vc3R5bGVzL3RoZW1lJztcclxuXHJcbmNvbnN0IGtleWZyYW1lX2Zvcl91cGRhdGVzX3Bsb3RzID0ga2V5ZnJhbWVzYFxyXG4gIDAlIHtcclxuICAgIGJhY2tncm91bmQ6ICR7dGhlbWUuY29sb3JzLnNlY29uZGFyeS5tYWlufTtcclxuICAgIGNvbG9yOiAgJHt0aGVtZS5jb2xvcnMuY29tbW9uLndoaXRlfTtcclxuICB9XHJcbiAgMTAwJSB7XHJcbiAgICBiYWNrZ3JvdW5kOiAke3RoZW1lLmNvbG9ycy5wcmltYXJ5LmxpZ2h0fTtcclxuICB9XHJcbmA7XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IFBhcmVudFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgc2l6ZTogU2l6ZVByb3BzLCBpc0xvYWRpbmc6IHN0cmluZywgYW5pbWF0aW9uOiBzdHJpbmcsIGlzUGxvdFNlbGVjdGVkPzogc3RyaW5nIH0+YFxyXG4gICAgd2lkdGg6ICR7KHByb3BzKSA9PiAocHJvcHMuc2l6ZS53ICsgMjQpfXB4O1xyXG4gICAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7XHJcbiAgICBtYXJnaW46IDRweDtcclxuICAgIGJhY2tncm91bmQ6ICR7KHByb3BzKSA9PiBwcm9wcy5pc1Bsb3RTZWxlY3RlZCA9PT0gJ3RydWUnID8gdGhlbWUuY29sb3JzLnNlY29uZGFyeS5saWdodCA6IHRoZW1lLmNvbG9ycy5wcmltYXJ5LmxpZ2h0fTtcclxuICAgIGRpc3BsYXk6IGdyaWQ7XHJcbiAgICBhbGlnbi1pdGVtczogZW5kO1xyXG4gICAgcGFkZGluZzogOHB4O1xyXG4gICAgYW5pbWF0aW9uLWl0ZXJhdGlvbi1jb3VudDogMTtcclxuICAgIGFuaW1hdGlvbi1kdXJhdGlvbjogMXM7XHJcbiAgICBhbmltYXRpb24tbmFtZTogJHsocHJvcHMpID0+XHJcbiAgICBwcm9wcy5pc0xvYWRpbmcgPT09ICd0cnVlJyAmJiBwcm9wcy5hbmltYXRpb24gPT09ICd0cnVlJ1xyXG4gICAgICA/IGtleWZyYW1lX2Zvcl91cGRhdGVzX3Bsb3RzXHJcbiAgICAgIDogJyd9O1xyXG5gXHJcblxyXG5leHBvcnQgY29uc3QgTGF5b3V0TmFtZSA9IHN0eWxlZC5kaXY8eyBlcnJvcj86IHN0cmluZywgaXNQbG90U2VsZWN0ZWQ/OiBzdHJpbmcgfT5gXHJcbiAgICBwYWRkaW5nLWJvdHRvbTogNDtcclxuICAgIGNvbG9yOiAke3Byb3BzID0+IHByb3BzLmVycm9yID09PSAndHJ1ZScgPyB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLmVycm9yIDogdGhlbWUuY29sb3JzLmNvbW1vbi5ibGFja307XHJcbiAgICBmb250LXdlaWdodDogJHtwcm9wcyA9PiBwcm9wcy5pc1Bsb3RTZWxlY3RlZCA9PT0gJ3RydWUnID8gJ2JvbGQnIDogJyd9O1xyXG4gICAgd29yZC1icmVhazogYnJlYWstd29yZDtcclxuYFxyXG5leHBvcnQgY29uc3QgTGF5b3V0V3JhcHBlciA9IHN0eWxlZC5kaXY8eyBzaXplOiBTaXplUHJvcHMgJiBzdHJpbmcsIGF1dG86IHN0cmluZyB9PmBcclxuICAgIC8vIHdpZHRoOiAkeyhwcm9wcykgPT4gcHJvcHMuc2l6ZS53ID8gYCR7cHJvcHMuc2l6ZS53ICsgMTJ9cHhgIDogcHJvcHMuc2l6ZX07XHJcbiAgICAvLyBoZWlnaHQ6JHsocHJvcHMpID0+IHByb3BzLnNpemUuaCA/IGAke3Byb3BzLnNpemUudyArIDE2fXB4YCA6IHByb3BzLnNpemV9O1xyXG4gICAgZGlzcGxheTogZ3JpZDtcclxuICAgIGdyaWQtdGVtcGxhdGUtY29sdW1uczogJHsocHJvcHMpID0+IChwcm9wcy5hdXRvKX07XHJcbiAgICBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjtcclxuYDtcclxuXHJcbmV4cG9ydCBjb25zdCBQbG90V3JhcHBlciA9IHN0eWxlZC5kaXY8eyBwbG90U2VsZWN0ZWQ6IGJvb2xlYW4sIHdpZHRoOiBzdHJpbmcsIGhlaWdodDogc3RyaW5nIH0+YFxyXG4gICAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7XHJcbiAgICBib3JkZXI6ICR7KHByb3BzKSA9PiBwcm9wcy5wbG90U2VsZWN0ZWQgPyBgNHB4IHNvbGlkICR7dGhlbWUuY29sb3JzLnNlY29uZGFyeS5saWdodH1gIDogYDJweCBzb2xpZCAke3RoZW1lLmNvbG9ycy5wcmltYXJ5LmxpZ2h0fWB9O1xyXG4gICAgYWxpZ24taXRlbXM6ICBjZW50ZXIgO1xyXG4gICAgd2lkdGg6ICR7KHByb3BzKSA9PiBwcm9wcy53aWR0aCA/IGBjYWxjKCR7cHJvcHMud2lkdGh9KzhweClgIDogJ2ZpdC1jb250ZW50J307XHJcbiAgICBoZWlnaHQ6ICR7KHByb3BzKSA9PiBwcm9wcy5oZWlnaHQgPyBwcm9wcy5oZWlnaHQgOiAnZml0LWNvbnRlbnQnfTs7XHJcbiAgICBjdXJzb3I6ICBwb2ludGVyIDtcclxuICAgIHBhZGRpbmc6IDRweDtcclxuICAgIGFsaWduLXNlbGY6ICBjZW50ZXIgO1xyXG4gICAganVzdGlmeS1zZWxmOiAgYmFzZWxpbmU7XHJcbiAgICBjdXJzb3I6ICR7cHJvcHMgPT4gcHJvcHMucGxvdFNlbGVjdGVkID8gJ3pvb20tb3V0JyA6ICd6b29tLWluJ307XHJcbmAiXSwic291cmNlUm9vdCI6IiJ9
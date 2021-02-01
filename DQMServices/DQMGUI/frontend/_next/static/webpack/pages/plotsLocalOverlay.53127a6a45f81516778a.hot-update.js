webpackHotUpdate_N_E("pages/plotsLocalOverlay",{

/***/ "./components/viewDetailsMenu/styledComponents.tsx":
/*!*********************************************************!*\
  !*** ./components/viewDetailsMenu/styledComponents.tsx ***!
  \*********************************************************/
/*! exports provided: CheckboxesWrapper, StyledDiv, ResultsWrapper, NavWrapper, StyledModal, FullWidthRow, StyledSelect, StyledCollapse, OptionParagraph, SelectedRunsTable, SelectedRunsTr, SelectedRunsTh, SelectedRunsTd */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "CheckboxesWrapper", function() { return CheckboxesWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "StyledDiv", function() { return StyledDiv; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ResultsWrapper", function() { return ResultsWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "NavWrapper", function() { return NavWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "StyledModal", function() { return StyledModal; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "FullWidthRow", function() { return FullWidthRow; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "StyledSelect", function() { return StyledSelect; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "StyledCollapse", function() { return StyledCollapse; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OptionParagraph", function() { return OptionParagraph; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SelectedRunsTable", function() { return SelectedRunsTable; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SelectedRunsTr", function() { return SelectedRunsTr; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SelectedRunsTh", function() { return SelectedRunsTh; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SelectedRunsTd", function() { return SelectedRunsTd; });
/* harmony import */ var styled_components__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! styled-components */ "./node_modules/styled-components/dist/styled-components.browser.esm.js");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");




var CheckboxesWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__CheckboxesWrapper",
  componentId: "sc-7cwei9-0"
})(["padding:calc(", "*2);"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].space.spaceBetween);
var StyledDiv = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__StyledDiv",
  componentId: "sc-7cwei9-1"
})(["display:flex;"]);
var ResultsWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__ResultsWrapper",
  componentId: "sc-7cwei9-2"
})(["overflow-x:hidden;height:60vh;width:fit-content;padding-top:calc(", "*2);width:auto;"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].space.padding);
var NavWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__NavWrapper",
  componentId: "sc-7cwei9-3"
})(["width:25vw;"]);
var StyledModal = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["default"])(antd__WEBPACK_IMPORTED_MODULE_1__["Modal"]).withConfig({
  displayName: "styledComponents__StyledModal",
  componentId: "sc-7cwei9-4"
})([".ant-modal-content{width:fit-content;};.ant-modal-body{width:max-content;}"]);
var FullWidthRow = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["default"])(antd__WEBPACK_IMPORTED_MODULE_1__["Row"]).withConfig({
  displayName: "styledComponents__FullWidthRow",
  componentId: "sc-7cwei9-5"
})(["width:100%;padding:", ";"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].space.spaceBetween);
var StyledSelect = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["default"])(antd__WEBPACK_IMPORTED_MODULE_1__["Select"]).withConfig({
  displayName: "styledComponents__StyledSelect",
  componentId: "sc-7cwei9-6"
})([".ant-select-selector{border-radius:12px !important;width:", " !important;font-weight:", " !important;}"], function (props) {
  return props.width ? "".concat(props.width) : 'fit-content';
}, function (props) {
  return props.selected === 'selected' ? 'bold' : 'inherit';
});
var StyledCollapse = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["default"])(antd__WEBPACK_IMPORTED_MODULE_1__["Collapse"]).withConfig({
  displayName: "styledComponents__StyledCollapse",
  componentId: "sc-7cwei9-7"
})(["width:100%;.ant-collapse-content > .ant-collapse-content-box{padding:", ";}"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].space.spaceBetween);
var OptionParagraph = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__OptionParagraph",
  componentId: "sc-7cwei9-8"
})(["display:flex;align-items:center;justify-content:center;width:100%;"]);
var SelectedRunsTable = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].table.withConfig({
  displayName: "styledComponents__SelectedRunsTable",
  componentId: "sc-7cwei9-9"
})(["text-align:center;"]);
var SelectedRunsTr = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].tr.withConfig({
  displayName: "styledComponents__SelectedRunsTr",
  componentId: "sc-7cwei9-10"
})(["border:1px solid ", ";"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].colors.primary.main);
var SelectedRunsTh = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].th.withConfig({
  displayName: "styledComponents__SelectedRunsTh",
  componentId: "sc-7cwei9-11"
})(["width:30%;border-right:1px solid ", ";padding:4px;background:", ";"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].colors.primary.main, _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].colors.primary.light);
var SelectedRunsTd = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].td.withConfig({
  displayName: "styledComponents__SelectedRunsTd",
  componentId: "sc-7cwei9-12"
})(["border-right:1px solid ", ";padding:4px;"], _styles_theme__WEBPACK_IMPORTED_MODULE_2__["theme"].colors.primary.main);

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cy50c3giXSwibmFtZXMiOlsiQ2hlY2tib3hlc1dyYXBwZXIiLCJzdHlsZWQiLCJkaXYiLCJ0aGVtZSIsInNwYWNlIiwic3BhY2VCZXR3ZWVuIiwiU3R5bGVkRGl2IiwiUmVzdWx0c1dyYXBwZXIiLCJwYWRkaW5nIiwiTmF2V3JhcHBlciIsIlN0eWxlZE1vZGFsIiwiTW9kYWwiLCJGdWxsV2lkdGhSb3ciLCJSb3ciLCJTdHlsZWRTZWxlY3QiLCJTZWxlY3QiLCJwcm9wcyIsIndpZHRoIiwic2VsZWN0ZWQiLCJTdHlsZWRDb2xsYXBzZSIsIkNvbGxhcHNlIiwiT3B0aW9uUGFyYWdyYXBoIiwiU2VsZWN0ZWRSdW5zVGFibGUiLCJ0YWJsZSIsIlNlbGVjdGVkUnVuc1RyIiwidHIiLCJjb2xvcnMiLCJwcmltYXJ5IiwibWFpbiIsIlNlbGVjdGVkUnVuc1RoIiwidGgiLCJsaWdodCIsIlNlbGVjdGVkUnVuc1RkIiwidGQiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFFTyxJQUFNQSxpQkFBaUIsR0FBR0MseURBQU0sQ0FBQ0MsR0FBVjtBQUFBO0FBQUE7QUFBQSw4QkFDWkMsbURBQUssQ0FBQ0MsS0FBTixDQUFZQyxZQURBLENBQXZCO0FBSUEsSUFBTUMsU0FBUyxHQUFHTCx5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLHFCQUFmO0FBSUEsSUFBTUssY0FBYyxHQUFHTix5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLDZGQUlMQyxtREFBSyxDQUFDQyxLQUFOLENBQVlJLE9BSlAsQ0FBcEI7QUFPQSxJQUFNQyxVQUFVLEdBQUdSLHlEQUFNLENBQUNDLEdBQVY7QUFBQTtBQUFBO0FBQUEsbUJBQWhCO0FBSUEsSUFBTVEsV0FBVyxHQUFHVCxpRUFBTSxDQUFDVSwwQ0FBRCxDQUFUO0FBQUE7QUFBQTtBQUFBLGtGQUFqQjtBQVNBLElBQU1DLFlBQVksR0FBR1gsaUVBQU0sQ0FBQ1ksd0NBQUQsQ0FBVDtBQUFBO0FBQUE7QUFBQSxpQ0FFWlYsbURBQUssQ0FBQ0MsS0FBTixDQUFZQyxZQUZBLENBQWxCO0FBSUEsSUFBTVMsWUFBWSxHQUFHYixpRUFBTSxDQUFDYywyQ0FBRCxDQUFUO0FBQUE7QUFBQTtBQUFBLCtHQU1aLFVBQUNDLEtBQUQ7QUFBQSxTQUFZQSxLQUFLLENBQUNDLEtBQU4sYUFBaUJELEtBQUssQ0FBQ0MsS0FBdkIsSUFBaUMsYUFBN0M7QUFBQSxDQU5ZLEVBT04sVUFBQ0QsS0FBRDtBQUFBLFNBQ2JBLEtBQUssQ0FBQ0UsUUFBTixLQUFtQixVQUFuQixHQUFnQyxNQUFoQyxHQUF5QyxTQUQ1QjtBQUFBLENBUE0sQ0FBbEI7QUFZQSxJQUFNQyxjQUFjLEdBQUdsQixpRUFBTSxDQUFDbUIsNkNBQUQsQ0FBVDtBQUFBO0FBQUE7QUFBQSxvRkFHWmpCLG1EQUFLLENBQUNDLEtBQU4sQ0FBWUMsWUFIQSxDQUFwQjtBQU1BLElBQU1nQixlQUFlLEdBQUdwQix5REFBTSxDQUFDQyxHQUFWO0FBQUE7QUFBQTtBQUFBLDBFQUFyQjtBQU9BLElBQU1vQixpQkFBaUIsR0FBR3JCLHlEQUFNLENBQUNzQixLQUFWO0FBQUE7QUFBQTtBQUFBLDBCQUF2QjtBQUdBLElBQU1DLGNBQWMsR0FBR3ZCLHlEQUFNLENBQUN3QixFQUFWO0FBQUE7QUFBQTtBQUFBLCtCQUNMdEIsbURBQUssQ0FBQ3VCLE1BQU4sQ0FBYUMsT0FBYixDQUFxQkMsSUFEaEIsQ0FBcEI7QUFHQSxJQUFNQyxjQUFjLEdBQUc1Qix5REFBTSxDQUFDNkIsRUFBVjtBQUFBO0FBQUE7QUFBQSwyRUFFQzNCLG1EQUFLLENBQUN1QixNQUFOLENBQWFDLE9BQWIsQ0FBcUJDLElBRnRCLEVBSVh6QixtREFBSyxDQUFDdUIsTUFBTixDQUFhQyxPQUFiLENBQXFCSSxLQUpWLENBQXBCO0FBTUEsSUFBTUMsY0FBYyxHQUFHL0IseURBQU0sQ0FBQ2dDLEVBQVY7QUFBQTtBQUFBO0FBQUEsaURBQ0M5QixtREFBSyxDQUFDdUIsTUFBTixDQUFhQyxPQUFiLENBQXFCQyxJQUR0QixDQUFwQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9wbG90c0xvY2FsT3ZlcmxheS41MzEyN2E2YTQ1ZjgxNTE2Nzc4YS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHN0eWxlZCBmcm9tICdzdHlsZWQtY29tcG9uZW50cyc7XHJcbmltcG9ydCB7IENvbGxhcHNlIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IE1vZGFsLCBSb3csIFNlbGVjdCB9IGZyb20gJ2FudGQnO1xyXG5cclxuZXhwb3J0IGNvbnN0IENoZWNrYm94ZXNXcmFwcGVyID0gc3R5bGVkLmRpdmBcclxuICBwYWRkaW5nOiBjYWxjKCR7dGhlbWUuc3BhY2Uuc3BhY2VCZXR3ZWVufSoyKTtcclxuYDtcclxuXHJcbmV4cG9ydCBjb25zdCBTdHlsZWREaXYgPSBzdHlsZWQuZGl2YFxyXG4gIGRpc3BsYXk6IGZsZXg7XHJcbmA7XHJcblxyXG5leHBvcnQgY29uc3QgUmVzdWx0c1dyYXBwZXIgPSBzdHlsZWQuZGl2YFxyXG4gIG92ZXJmbG93LXg6IGhpZGRlbjtcclxuICBoZWlnaHQ6IDYwdmg7XHJcbiAgd2lkdGg6IGZpdC1jb250ZW50O1xyXG4gIHBhZGRpbmctdG9wOiBjYWxjKCR7dGhlbWUuc3BhY2UucGFkZGluZ30qMik7XHJcbiAgd2lkdGg6IGF1dG87XHJcbmA7XHJcbmV4cG9ydCBjb25zdCBOYXZXcmFwcGVyID0gc3R5bGVkLmRpdmBcclxuICB3aWR0aDogMjV2dztcclxuYDtcclxuXHJcbmV4cG9ydCBjb25zdCBTdHlsZWRNb2RhbCA9IHN0eWxlZChNb2RhbCk8eyB3aWR0aD86IHN0cmluZyB9PmBcclxuICAuYW50LW1vZGFsLWNvbnRlbnQge1xyXG4gICAgd2lkdGg6IGZpdC1jb250ZW50O1xyXG4gIH07XHJcbiAgLmFudC1tb2RhbC1ib2R5e1xyXG4gICAgd2lkdGg6IG1heC1jb250ZW50O1xyXG4gIH1cclxuYDtcclxuXHJcbmV4cG9ydCBjb25zdCBGdWxsV2lkdGhSb3cgPSBzdHlsZWQoUm93KWBcclxuICB3aWR0aDogMTAwJTtcclxuICBwYWRkaW5nOiAke3RoZW1lLnNwYWNlLnNwYWNlQmV0d2Vlbn07XHJcbmA7XHJcbmV4cG9ydCBjb25zdCBTdHlsZWRTZWxlY3QgPSBzdHlsZWQoU2VsZWN0KTx7XHJcbiAgc2VsZWN0ZWQ/OiBzdHJpbmc7XHJcbiAgd2lkdGg/OiBzdHJpbmcgfCB1bmRlZmluZWQ7XHJcbn0+YFxyXG4gIC5hbnQtc2VsZWN0LXNlbGVjdG9yIHtcclxuICAgIGJvcmRlci1yYWRpdXM6IDEycHggIWltcG9ydGFudDtcclxuICAgIHdpZHRoOiAkeyhwcm9wcykgPT4gKHByb3BzLndpZHRoID8gYCR7cHJvcHMud2lkdGh9YCA6ICdmaXQtY29udGVudCcpfSAhaW1wb3J0YW50O1xyXG4gICAgZm9udC13ZWlnaHQ6ICR7KHByb3BzKSA9PlxyXG4gICAgICBwcm9wcy5zZWxlY3RlZCA9PT0gJ3NlbGVjdGVkJyA/ICdib2xkJyA6ICdpbmhlcml0J30gIWltcG9ydGFudDtcclxuICB9XHJcbmA7XHJcblxyXG5leHBvcnQgY29uc3QgU3R5bGVkQ29sbGFwc2UgPSBzdHlsZWQoQ29sbGFwc2UpYFxyXG4gIHdpZHRoOiAxMDAlO1xyXG4gIC5hbnQtY29sbGFwc2UtY29udGVudCA+IC5hbnQtY29sbGFwc2UtY29udGVudC1ib3gge1xyXG4gICAgcGFkZGluZzogJHt0aGVtZS5zcGFjZS5zcGFjZUJldHdlZW59O1xyXG4gIH1cclxuYDtcclxuZXhwb3J0IGNvbnN0IE9wdGlvblBhcmFncmFwaCA9IHN0eWxlZC5kaXZgXHJcbiAgZGlzcGxheTogZmxleDtcclxuICBhbGlnbi1pdGVtczogY2VudGVyO1xyXG4gIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG4gIHdpZHRoOiAxMDAlO1xyXG5gO1xyXG5cclxuZXhwb3J0IGNvbnN0IFNlbGVjdGVkUnVuc1RhYmxlID0gc3R5bGVkLnRhYmxlYFxyXG4gIHRleHQtYWxpZ246IGNlbnRlcjtcclxuYDtcclxuZXhwb3J0IGNvbnN0IFNlbGVjdGVkUnVuc1RyID0gc3R5bGVkLnRyYFxyXG4gIGJvcmRlcjogMXB4IHNvbGlkICR7dGhlbWUuY29sb3JzLnByaW1hcnkubWFpbn07XHJcbmA7XHJcbmV4cG9ydCBjb25zdCBTZWxlY3RlZFJ1bnNUaCA9IHN0eWxlZC50aGBcclxuICB3aWR0aDogMzAlO1xyXG4gIGJvcmRlci1yaWdodDogMXB4IHNvbGlkICR7dGhlbWUuY29sb3JzLnByaW1hcnkubWFpbn07XHJcbiAgcGFkZGluZzogNHB4O1xyXG4gIGJhY2tncm91bmQ6ICR7dGhlbWUuY29sb3JzLnByaW1hcnkubGlnaHR9O1xyXG5gO1xyXG5leHBvcnQgY29uc3QgU2VsZWN0ZWRSdW5zVGQgPSBzdHlsZWQudGRgXHJcbiAgYm9yZGVyLXJpZ2h0OiAxcHggc29saWQgJHt0aGVtZS5jb2xvcnMucHJpbWFyeS5tYWlufTtcclxuICBwYWRkaW5nOiA0cHg7XHJcbmA7XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
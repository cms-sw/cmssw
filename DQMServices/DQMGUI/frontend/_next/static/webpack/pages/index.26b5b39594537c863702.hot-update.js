webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotsWithLayouts/plot.tsx":
/*!*********************************************************!*\
  !*** ./components/plots/plot/plotsWithLayouts/plot.tsx ***!
  \*********************************************************/
/*! exports provided: Plot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Plot", function() { return Plot; });
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./styledComponents */ "./components/plots/plot/plotsWithLayouts/styledComponents.ts");
/* harmony import */ var _singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _plotImage__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../plotImage */ "./components/plots/plot/plotImage.tsx");



var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/plot/plotsWithLayouts/plot.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];







var Plot = function Plot(_ref) {
  var globalState = _ref.globalState,
      query = _ref.query,
      plot = _ref.plot,
      onePlotHeight = _ref.onePlotHeight,
      onePlotWidth = _ref.onePlotWidth,
      selected_plots = _ref.selected_plots,
      imageRef = _ref.imageRef,
      imageRefScrollDown = _ref.imageRefScrollDown,
      blink = _ref.blink,
      updated_by_not_older_than = _ref.updated_by_not_older_than;
  var params_for_api = Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["FormatParamsForAPI"])(globalState, query, encodeURI(plot.name), plot.path);
  params_for_api.width = onePlotWidth;
  params_for_api.height = onePlotHeight;
  var url = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_url"])(params_for_api);
  var overlaid_plots_urls = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_overlaied_plots_urls"])(params_for_api);
  var joined_overlaid_plots_urls = overlaid_plots_urls.join('');
  params_for_api.joined_overlaied_plots_urls = joined_overlaid_plots_urls;
  var plot_with_overlay = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_with_overlay"])(params_for_api);
  plot.dataset_name = query.dataset_name;
  plot.run_number = query.run_number;
  var plotSelected = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["isPlotSelected"])(selected_plots, plot);
  var fullPlotPath = plot.path + '/' + plot.name;
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: fullPlotPath,
    color: Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["get_plot_error"])(plot) ? 'red' : '',
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_6__["PlotWrapper"], {
    height: "".concat(onePlotHeight, "px"),
    width: "".concat(onePlotWidth, "px"),
    plotSelected: plotSelected,
    onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
      return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
        while (1) {
          switch (_context.prev = _context.next) {
            case 0:
              _context.next = 2;
              return plotSelected;

            case 2:
              setTimeout(function () {
                Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["scroll"])(imageRef);
                Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["scrollToBottom"])(imageRefScrollDown);
              }, 500);

            case 3:
            case "end":
              return _context.stop();
          }
        }
      }, _callee);
    })),
    ref: imageRef,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 7
    }
  }, query.overlay_data ? __jsx(_plotImage__WEBPACK_IMPORTED_MODULE_8__["PlotImage"], {
    blink: blink,
    params_for_api: params_for_api,
    plot: plot,
    plotURL: plot_with_overlay,
    updated_by_not_older_than: updated_by_not_older_than,
    query: query,
    imageRef: imageRef,
    isPlotSelected: plotSelected,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 11
    }
  }) : __jsx(_plotImage__WEBPACK_IMPORTED_MODULE_8__["PlotImage"], {
    blink: blink,
    params_for_api: params_for_api,
    plot: plot,
    plotURL: url,
    updated_by_not_older_than: updated_by_not_older_than,
    query: query,
    imageRef: imageRef,
    isPlotSelected: plotSelected,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 12
    }
  })));
};
_c = Plot;

var _c;

$RefreshReg$(_c, "Plot");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvcGxvdC50c3giXSwibmFtZXMiOlsiUGxvdCIsImdsb2JhbFN0YXRlIiwicXVlcnkiLCJwbG90Iiwib25lUGxvdEhlaWdodCIsIm9uZVBsb3RXaWR0aCIsInNlbGVjdGVkX3Bsb3RzIiwiaW1hZ2VSZWYiLCJpbWFnZVJlZlNjcm9sbERvd24iLCJibGluayIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJwYXJhbXNfZm9yX2FwaSIsIkZvcm1hdFBhcmFtc0ZvckFQSSIsImVuY29kZVVSSSIsIm5hbWUiLCJwYXRoIiwid2lkdGgiLCJoZWlnaHQiLCJ1cmwiLCJnZXRfcGxvdF91cmwiLCJvdmVybGFpZF9wbG90c191cmxzIiwiZ2V0X292ZXJsYWllZF9wbG90c191cmxzIiwiam9pbmVkX292ZXJsYWlkX3Bsb3RzX3VybHMiLCJqb2luIiwiam9pbmVkX292ZXJsYWllZF9wbG90c191cmxzIiwicGxvdF93aXRoX292ZXJsYXkiLCJnZXRfcGxvdF93aXRoX292ZXJsYXkiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwicGxvdFNlbGVjdGVkIiwiaXNQbG90U2VsZWN0ZWQiLCJmdWxsUGxvdFBhdGgiLCJnZXRfcGxvdF9lcnJvciIsInNldFRpbWVvdXQiLCJzY3JvbGwiLCJzY3JvbGxUb0JvdHRvbSIsIm92ZXJsYXlfZGF0YSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFlTyxJQUFNQSxJQUFJLEdBQUcsU0FBUEEsSUFBTyxPQVV5QjtBQUFBLE1BVDNDQyxXQVMyQyxRQVQzQ0EsV0FTMkM7QUFBQSxNQVIzQ0MsS0FRMkMsUUFSM0NBLEtBUTJDO0FBQUEsTUFQM0NDLElBTzJDLFFBUDNDQSxJQU8yQztBQUFBLE1BTjNDQyxhQU0yQyxRQU4zQ0EsYUFNMkM7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QVFDLGtCQUFjLEVBQUVzQixZQVJqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBMUJMLENBREYsQ0FERjtBQXlDRCxDQXhFTTtLQUFNN0IsSSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4yNmI1YjM5NTk0NTM3Yzg2MzcwMi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnXHJcbmltcG9ydCB7IFRvb2x0aXAgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscywgZ2V0X3Bsb3RfdXJsLCBnZXRfcGxvdF93aXRoX292ZXJsYXkgfSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHsgUGxvdERhdGFQcm9wcywgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgaXNQbG90U2VsZWN0ZWQgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQgeyBQbG90V3JhcHBlciB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSwgZ2V0X3Bsb3RfZXJyb3IsIHNjcm9sbCwgc2Nyb2xsVG9Cb3R0b20gfSBmcm9tICcuLi9zaW5nbGVQbG90L3V0aWxzJ1xyXG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi9wbG90SW1hZ2UnO1xyXG5cclxuaW50ZXJmYWNlIFBsb3RQcm9wcyB7XHJcbiAgZ2xvYmFsU3RhdGU6IGFueTtcclxuICBxdWVyeTogUXVlcnlQcm9wcztcclxuICBwbG90OiBQbG90RGF0YVByb3BzO1xyXG4gIG9uZVBsb3RXaWR0aDogbnVtYmVyO1xyXG4gIG9uZVBsb3RIZWlnaHQ6IG51bWJlcjtcclxuICBzZWxlY3RlZF9wbG90czogUGxvdERhdGFQcm9wc1tdO1xyXG4gIGltYWdlUmVmOiBSZWFjdC5SZWZPYmplY3Q8SFRNTERpdkVsZW1lbnQ+O1xyXG4gIGltYWdlUmVmU2Nyb2xsRG93bjogUmVhY3QuUmVmT2JqZWN0PEhUTUxEaXZFbGVtZW50PjtcclxuICBibGluazogYm9vbGVhbjtcclxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuOiBudW1iZXI7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBQbG90ID0gKHtcclxuICBnbG9iYWxTdGF0ZSxcclxuICBxdWVyeSxcclxuICBwbG90LFxyXG4gIG9uZVBsb3RIZWlnaHQsXHJcbiAgb25lUGxvdFdpZHRoLFxyXG4gIHNlbGVjdGVkX3Bsb3RzLFxyXG4gIGltYWdlUmVmLFxyXG4gIGltYWdlUmVmU2Nyb2xsRG93bixcclxuICBibGluayxcclxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH06IFBsb3RQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHBhcmFtc19mb3JfYXBpID0gRm9ybWF0UGFyYW1zRm9yQVBJKFxyXG4gICAgZ2xvYmFsU3RhdGUsXHJcbiAgICBxdWVyeSxcclxuICAgIGVuY29kZVVSSShwbG90Lm5hbWUpLFxyXG4gICAgcGxvdC5wYXRoXHJcbiAgKTtcclxuICBwYXJhbXNfZm9yX2FwaS53aWR0aCA9IG9uZVBsb3RXaWR0aFxyXG4gIHBhcmFtc19mb3JfYXBpLmhlaWdodCA9IG9uZVBsb3RIZWlnaHRcclxuICBjb25zdCB1cmwgPSBnZXRfcGxvdF91cmwocGFyYW1zX2Zvcl9hcGkpO1xyXG4gIGNvbnN0IG92ZXJsYWlkX3Bsb3RzX3VybHMgPSBnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMocGFyYW1zX2Zvcl9hcGkpO1xyXG4gIGNvbnN0IGpvaW5lZF9vdmVybGFpZF9wbG90c191cmxzID0gb3ZlcmxhaWRfcGxvdHNfdXJscy5qb2luKCcnKTtcclxuICBwYXJhbXNfZm9yX2FwaS5qb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMgPSBqb2luZWRfb3ZlcmxhaWRfcGxvdHNfdXJscztcclxuICBjb25zdCBwbG90X3dpdGhfb3ZlcmxheSA9IGdldF9wbG90X3dpdGhfb3ZlcmxheShwYXJhbXNfZm9yX2FwaSk7XHJcbiAgcGxvdC5kYXRhc2V0X25hbWUgPSBxdWVyeS5kYXRhc2V0X25hbWVcclxuICBwbG90LnJ1bl9udW1iZXIgPSBxdWVyeS5ydW5fbnVtYmVyXHJcbiAgY29uc3QgcGxvdFNlbGVjdGVkID0gaXNQbG90U2VsZWN0ZWQoXHJcbiAgICBzZWxlY3RlZF9wbG90cyxcclxuICAgIHBsb3RcclxuICApXHJcbiAgY29uc3QgZnVsbFBsb3RQYXRoID0gcGxvdC5wYXRoICsgJy8nICsgcGxvdC5uYW1lXHJcbiAgcmV0dXJuIChcclxuICAgIDxUb29sdGlwIHRpdGxlPXtmdWxsUGxvdFBhdGh9IGNvbG9yPXtnZXRfcGxvdF9lcnJvcihwbG90KSA/ICdyZWQnIDogJyd9PlxyXG4gICAgICA8UGxvdFdyYXBwZXJcclxuICAgICAgICBoZWlnaHQ9e2Ake29uZVBsb3RIZWlnaHR9cHhgfVxyXG4gICAgICAgIHdpZHRoPXtgJHtvbmVQbG90V2lkdGh9cHhgfVxyXG4gICAgICAgIHBsb3RTZWxlY3RlZD17cGxvdFNlbGVjdGVkfVxyXG4gICAgICAgIG9uQ2xpY2s9e2FzeW5jICgpID0+IHtcclxuICAgICAgICAgIGF3YWl0IHBsb3RTZWxlY3RlZFxyXG4gICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XHJcbiAgICAgICAgICAgIHNjcm9sbChpbWFnZVJlZik7XHJcbiAgICAgICAgICAgIHNjcm9sbFRvQm90dG9tKGltYWdlUmVmU2Nyb2xsRG93bilcclxuICAgICAgICAgIH0sIDUwMCk7XHJcbiAgICAgICAgfX1cclxuXHJcbiAgICAgICAgcmVmPXtpbWFnZVJlZn1cclxuICAgICAgPlxyXG4gICAgICAgIHtxdWVyeS5vdmVybGF5X2RhdGEgPyAoXHJcbiAgICAgICAgICA8UGxvdEltYWdlXHJcbiAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e3BhcmFtc19mb3JfYXBpfVxyXG4gICAgICAgICAgICBwbG90PXtwbG90fVxyXG4gICAgICAgICAgICBwbG90VVJMPXtwbG90X3dpdGhfb3ZlcmxheX1cclxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICBpbWFnZVJlZj17aW1hZ2VSZWZ9XHJcbiAgICAgICAgICAgIGlzUGxvdFNlbGVjdGVkPXtwbG90U2VsZWN0ZWR9XHJcbiAgICAgICAgICAvPilcclxuICAgICAgICAgIDpcclxuICAgICAgICAgICg8UGxvdEltYWdlXHJcbiAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e3BhcmFtc19mb3JfYXBpfVxyXG4gICAgICAgICAgICBwbG90PXtwbG90fVxyXG4gICAgICAgICAgICBwbG90VVJMPXt1cmx9XHJcbiAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgICAgaW1hZ2VSZWY9e2ltYWdlUmVmfVxyXG4gICAgICAgICAgICBpc1Bsb3RTZWxlY3RlZD17cGxvdFNlbGVjdGVkfVxyXG4gICAgICAgICAgLz4pfVxyXG4gICAgICA8L1Bsb3RXcmFwcGVyPlxyXG4gICAgPC9Ub29sdGlwPlxyXG4gIClcclxufVxyXG5cclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
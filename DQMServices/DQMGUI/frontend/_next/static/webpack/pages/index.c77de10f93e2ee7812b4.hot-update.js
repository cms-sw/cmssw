webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidPlot.tsx":
/*!********************************************************************************!*\
  !*** ./components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidPlot.tsx ***!
  \********************************************************************************/
/*! exports provided: ZoomedOverlaidPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedOverlaidPlot", function() { return ZoomedOverlaidPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/defineProperty */ "./node_modules/@babel/runtime/helpers/esm/defineProperty.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./utils */ "./components/plots/zoomedPlots/zoomedOverlayPlots/utils.ts");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _menu__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../menu */ "./components/plots/zoomedPlots/menu.tsx");
/* harmony import */ var _customization__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../../customization */ "./components/customization/index.tsx");
/* harmony import */ var _containers_display_portal__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../../../containers/display/portal */ "./containers/display/portal/index.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");
/* harmony import */ var _plot_plotImage__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../plot/plotImage */ "./components/plots/plot/plotImage.tsx");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../../utils */ "./components/utils.ts");



var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidPlot.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__["default"])(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }














var ZoomedOverlaidPlot = function ZoomedOverlaidPlot(_ref) {
  _s();

  var _copy_of_params$width, _params_for_api$width;

  var selected_plot = _ref.selected_plot,
      params_for_api = _ref.params_for_api;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(),
      customizationParams = _useState[0],
      setCustomizationParams = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(false),
      openCustomization = _useState2[0],
      toggleCustomizationMenu = _useState2[1];

  params_for_api.customizeProps = customizationParams;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      isPortalWindowOpen = _React$useState2[0],
      setIsPortalWindowOpen = _React$useState2[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;
  var url = Object(_utils__WEBPACK_IMPORTED_MODULE_14__["getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames"])(router.basePath, query, selected_plot);
  var zoomedPlotMenuOptions = [{
    label: 'Open in a new tab',
    value: 'open_in_a_new_tab',
    action: function action() {
      return setIsPortalWindowOpen(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["FullscreenOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 62,
        columnNumber: 13
      }
    })
  }, {
    label: 'Customize',
    value: 'Customize',
    action: function action() {
      return toggleCustomizationMenu(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["SettingOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 68,
        columnNumber: 13
      }
    })
  }, _config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end && {
    label: 'Overlay with another plot',
    value: 'overlay',
    url: url,
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 74,
        columnNumber: 13
      }
    })
  }];
  var overlaid_plots_urls = Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_overlaied_plots_urls"])(params_for_api);
  var joined_overlaid_plots_urls = overlaid_plots_urls.join('');
  params_for_api.joined_overlaied_plots_urls = joined_overlaid_plots_urls;
  var source = Object(_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_source"])(params_for_api);

  var copy_of_params = _objectSpread({}, params_for_api);

  copy_of_params.height = window.innerHeight;
  copy_of_params.width = Math.round(window.innerHeight * 1.33);
  var zoomed_plot_url = Object(_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_source"])(copy_of_params);

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink,
      updated_by_not_older_than = _useBlinkOnUpdate.updated_by_not_older_than;

  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 92,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_11__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledPlotRow"], {
    justifycontent: "center",
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].mode === 'ONLINE').toString(),
    minheight: copy_of_params.height,
    width: (_copy_of_params$width = copy_of_params.width) === null || _copy_of_params$width === void 0 ? void 0 : _copy_of_params$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 107,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 110,
      columnNumber: 11
    }
  }, __jsx(_plot_plotImage__WEBPACK_IMPORTED_MODULE_13__["PlotImage"], {
    blink: blink,
    params_for_api: copy_of_params,
    plot: selected_plot,
    plotURL: zoomed_plot_url,
    query: query,
    updated_by_not_older_than: updated_by_not_older_than,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 115,
      columnNumber: 13
    }
  })))), __jsx(_customization__WEBPACK_IMPORTED_MODULE_10__["Customization"], {
    plot_name: selected_plot.name,
    open: openCustomization,
    onCancel: function onCancel() {
      return toggleCustomizationMenu(false);
    },
    setCustomizationParams: setCustomizationParams,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 126,
      columnNumber: 7
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledPlotRow"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].mode === 'ONLINE').toString(),
    minheight: params_for_api.height,
    width: (_params_for_api$width = params_for_api.width) === null || _params_for_api$width === void 0 ? void 0 : _params_for_api$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    justifycontent: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 132,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 141,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 144,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_9__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 145,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 146,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ImageDiv"], {
    id: selected_plot.name,
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 150,
      columnNumber: 9
    }
  }, __jsx(_plot_plotImage__WEBPACK_IMPORTED_MODULE_13__["PlotImage"], {
    blink: blink,
    params_for_api: params_for_api,
    plot: selected_plot,
    plotURL: source,
    query: query,
    updated_by_not_older_than: updated_by_not_older_than,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 155,
      columnNumber: 11
    }
  }))));
};

_s(ZoomedOverlaidPlot, "n7HfDH0SxZV5E2eKjp3X83/7eok=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"], _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__["useBlinkOnUpdate"]];
});

_c = ZoomedOverlaidPlot;

var _c;

$RefreshReg$(_c, "ZoomedOverlaidPlot");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRPdmVybGF5UGxvdHMvem9vbWVkT3ZlcmxhaWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRPdmVybGFpZFBsb3QiLCJzZWxlY3RlZF9wbG90IiwicGFyYW1zX2Zvcl9hcGkiLCJ1c2VTdGF0ZSIsImN1c3RvbWl6YXRpb25QYXJhbXMiLCJzZXRDdXN0b21pemF0aW9uUGFyYW1zIiwib3BlbkN1c3RvbWl6YXRpb24iLCJ0b2dnbGVDdXN0b21pemF0aW9uTWVudSIsImN1c3RvbWl6ZVByb3BzIiwiUmVhY3QiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInVybCIsImdldFpvb21lZE92ZXJsYWlkUGxvdHNVcmxGb3JPdmVybGF5aW5nUGxvdHNXaXRoRGlmZmVyZW50TmFtZXMiLCJiYXNlUGF0aCIsInpvb21lZFBsb3RNZW51T3B0aW9ucyIsImxhYmVsIiwidmFsdWUiLCJhY3Rpb24iLCJpY29uIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm5ld19iYWNrX2VuZCIsIm92ZXJsYWlkX3Bsb3RzX3VybHMiLCJnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMiLCJqb2luZWRfb3ZlcmxhaWRfcGxvdHNfdXJscyIsImpvaW4iLCJqb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMiLCJzb3VyY2UiLCJnZXRfcGxvdF9zb3VyY2UiLCJjb3B5X29mX3BhcmFtcyIsImhlaWdodCIsIndpbmRvdyIsImlubmVySGVpZ2h0Iiwid2lkdGgiLCJNYXRoIiwicm91bmQiLCJ6b29tZWRfcGxvdF91cmwiLCJ1c2VCbGlua09uVXBkYXRlIiwiYmxpbmsiLCJ1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIiwibmFtZSIsInRvU3RyaW5nIiwibW9kZSIsImdldF9wbG90X2Vycm9yIiwicmVtb3ZlUGxvdEZyb21SaWdo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cUJBQXpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixFQUVFLE1BQUMsOEVBQUQ7QUFDRSxXQUFPLEVBQUU7QUFBQSxhQUFNNkIsc0ZBQXVCLENBQUNqQyxLQUFELEVBQVFiLGFBQVIsQ0FBN0I7QUFBQSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFGRixDQVpGLEVBa0JFLE1BQUMsNkVBQUQ7QUFDRSxNQUFFLEVBQUVBLGFBQWEsQ0FBQzBDLElBRHBCO0FBRUUsU0FBSyxFQUFFekMsY0FBYyxDQUFDa0MsS0FGeEI7QUFHRSxVQUFNLEVBQUVsQyxjQUFjLENBQUMrQixNQUh6QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0UsTUFBQywwREFBRDtBQUNFLFNBQUssRUFBRVEsS0FEVDtBQUVFLGtCQUFjLEVBQUV2QyxjQUZsQjtBQUdFLFFBQUksRUFBRUQsYUFIUjtBQUlFLFdBQU8sRUFBRTZCLE1BSlg7QUFLRSxTQUFLLEVBQUVoQixLQUxUO0FBTUUsNkJBQXlCLEVBQUU0Qix5QkFON0I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUxGLENBbEJGLENBeENGLENBREY7QUE0RUQsQ0E3SE07O0dBQU0xQyxrQjtVQVdJYSxxRCxFQW9DOEIyQix5RTs7O0tBL0NsQ3hDLGtCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmM3N2RlMTBmOTNlMmVlNzgxMmI0LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgU3RvcmUgfSBmcm9tICdhbnRkL2xpYi9mb3JtL2ludGVyZmFjZSc7XHJcbmltcG9ydCB7IFNldHRpbmdPdXRsaW5lZCwgRnVsbHNjcmVlbk91dGxpbmVkLCBCbG9ja091dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHtcclxuICBnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMsXHJcbiAgZnVuY3Rpb25zX2NvbmZpZyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBQbG90RGF0YVByb3BzLFxyXG4gIFF1ZXJ5UHJvcHMsXHJcbiAgQ3VzdG9taXplUHJvcHMsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBnZXRfcGxvdF9zb3VyY2UgfSBmcm9tICcuL3V0aWxzJztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRQbG90Um93LFxyXG4gIFBsb3ROYW1lQ29sLFxyXG4gIENvbHVtbixcclxuICBTdHlsZWRDb2wsXHJcbiAgSW1hZ2VEaXYsXHJcbiAgSW1hZ2UsXHJcbiAgTWludXNJY29uLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcclxuICBnZXRfcGxvdF9lcnJvcixcclxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBab29tZWRQbG90TWVudSB9IGZyb20gJy4uL21lbnUnO1xyXG5pbXBvcnQgeyBDdXN0b21pemF0aW9uIH0gZnJvbSAnLi4vLi4vLi4vY3VzdG9taXphdGlvbic7XHJcbmltcG9ydCB7IFBsb3RfcG9ydGFsIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3BvcnRhbCc7XHJcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJztcclxuaW1wb3J0IHsgUGxvdEltYWdlIH0gZnJvbSAnLi4vLi4vcGxvdC9wbG90SW1hZ2UnO1xyXG5pbXBvcnQgeyBnZXRab29tZWRPdmVybGFpZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzIH0gZnJvbSAnLi4vLi4vLi4vdXRpbHMnO1xyXG5cclxuaW50ZXJmYWNlIFpvb21lZFBsb3RzUHJvcHMge1xyXG4gIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHM7XHJcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgWm9vbWVkT3ZlcmxhaWRQbG90ID0gKHtcclxuICBzZWxlY3RlZF9wbG90LFxyXG4gIHBhcmFtc19mb3JfYXBpLFxyXG59OiBab29tZWRQbG90c1Byb3BzKSA9PiB7XHJcbiAgY29uc3QgW2N1c3RvbWl6YXRpb25QYXJhbXMsIHNldEN1c3RvbWl6YXRpb25QYXJhbXNdID0gdXNlU3RhdGU8XHJcbiAgICBQYXJ0aWFsPFN0b3JlPiAmIEN1c3RvbWl6ZVByb3BzXHJcbiAgPigpO1xyXG4gIGNvbnN0IFtvcGVuQ3VzdG9taXphdGlvbiwgdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnVdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIHBhcmFtc19mb3JfYXBpLmN1c3RvbWl6ZVByb3BzID0gY3VzdG9taXphdGlvblBhcmFtcztcclxuICBjb25zdCBbaXNQb3J0YWxXaW5kb3dPcGVuLCBzZXRJc1BvcnRhbFdpbmRvd09wZW5dID0gUmVhY3QudXNlU3RhdGUoZmFsc2UpO1xyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCB1cmwgPSBnZXRab29tZWRPdmVybGFpZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzKHJvdXRlci5iYXNlUGF0aCwgcXVlcnksIHNlbGVjdGVkX3Bsb3QpXHJcblxyXG4gIGNvbnN0IHpvb21lZFBsb3RNZW51T3B0aW9ucyA9IFtcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdPcGVuIGluIGEgbmV3IHRhYicsXHJcbiAgICAgIHZhbHVlOiAnb3Blbl9pbl9hX25ld190YWInLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHNldElzUG9ydGFsV2luZG93T3Blbih0cnVlKSxcclxuICAgICAgaWNvbjogPEZ1bGxzY3JlZW5PdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgICB7XHJcbiAgICAgIGxhYmVsOiAnQ3VzdG9taXplJyxcclxuICAgICAgdmFsdWU6ICdDdXN0b21pemUnLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KHRydWUpLFxyXG4gICAgICBpY29uOiA8U2V0dGluZ091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCAmJiB7XHJcbiAgICAgIGxhYmVsOiAnT3ZlcmxheSB3aXRoIGFub3RoZXIgcGxvdCcsXHJcbiAgICAgIHZhbHVlOiAnb3ZlcmxheScsXHJcbiAgICAgIHVybDogdXJsLFxyXG4gICAgICBpY29uOiA8QmxvY2tPdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgXTtcclxuXHJcbiAgY29uc3Qgb3ZlcmxhaWRfcGxvdHNfdXJscyA9IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyhwYXJhbXNfZm9yX2FwaSk7XHJcbiAgY29uc3Qgam9pbmVkX292ZXJsYWlkX3Bsb3RzX3VybHMgPSBvdmVybGFpZF9wbG90c191cmxzLmpvaW4oJycpO1xyXG4gIHBhcmFtc19mb3JfYXBpLmpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJscyA9IGpvaW5lZF9vdmVybGFpZF9wbG90c191cmxzO1xyXG5cclxuICBjb25zdCBzb3VyY2UgPSBnZXRfcGxvdF9zb3VyY2UocGFyYW1zX2Zvcl9hcGkpO1xyXG5cclxuICBjb25zdCBjb3B5X29mX3BhcmFtcyA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XHJcbiAgY29weV9vZl9wYXJhbXMud2lkdGggPSBNYXRoLnJvdW5kKHdpbmRvdy5pbm5lckhlaWdodCAqIDEuMzMpO1xyXG4gIGNvbnN0IHpvb21lZF9wbG90X3VybCA9IGdldF9wbG90X3NvdXJjZShjb3B5X29mX3BhcmFtcyk7XHJcblxyXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICA8UGxvdF9wb3J0YWxcclxuICAgICAgICBpc1BvcnRhbFdpbmRvd09wZW49e2lzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICBzZXRJc1BvcnRhbFdpbmRvd09wZW49e3NldElzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICB0aXRsZT17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFN0eWxlZFBsb3RSb3dcclxuICAgICAgICAgIGp1c3RpZnljb250ZW50PVwiY2VudGVyXCJcclxuICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBtaW5oZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aH1cclxuICAgICAgICAgICAgaGVpZ2h0PXtjb3B5X29mX3BhcmFtcy5oZWlnaHR9XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e2NvcHlfb2ZfcGFyYW1zfVxyXG4gICAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgICAgcGxvdFVSTD17em9vbWVkX3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgICAvPlxyXG4gICAgICAgICAgPC9JbWFnZURpdj5cclxuICAgICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICAgIDwvUGxvdF9wb3J0YWw+XHJcbiAgICAgIDxDdXN0b21pemF0aW9uXHJcbiAgICAgICAgcGxvdF9uYW1lPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgb3Blbj17b3BlbkN1c3RvbWl6YXRpb259XHJcbiAgICAgICAgb25DYW5jZWw9eygpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KGZhbHNlKX1cclxuICAgICAgICBzZXRDdXN0b21pemF0aW9uUGFyYW1zPXtzZXRDdXN0b21pemF0aW9uUGFyYW1zfVxyXG4gICAgICAvPlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxyXG4gICAgICA+XHJcbiAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICA8Q29sdW1uIGRpc3BsYXk9XCJmbGV4XCI+XHJcbiAgICAgICAgICA8Wm9vbWVkUGxvdE1lbnUgb3B0aW9ucz17em9vbWVkUGxvdE1lbnVPcHRpb25zfSAvPlxyXG4gICAgICAgICAgPE1pbnVzSWNvblxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZShxdWVyeSwgc2VsZWN0ZWRfcGxvdCl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sdW1uPlxyXG4gICAgICAgIDxJbWFnZURpdlxyXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aH1cclxuICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XHJcbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgIHBsb3RVUkw9e3NvdXJjZX1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
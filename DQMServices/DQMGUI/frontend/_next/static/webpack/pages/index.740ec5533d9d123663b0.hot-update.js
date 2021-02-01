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
        lineNumber: 57,
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
        lineNumber: 63,
        columnNumber: 13
      }
    })
  }, _config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end && {
    label: 'Overlay with another plot',
    value: 'overlay',
    action: function action() {
      var _query$overlay_data;

      var basePath = router.basePath;
      var page = 'plotsLocalOverlay';
      var run = 'run_number=' + query.run_number;
      var dataset = 'dataset_name=' + query.dataset_name;
      var path = 'folders_path=' + selected_plot.path;
      var plot_name = 'plot_name=' + selected_plot.name;
      var globally_overlaid_plots = (_query$overlay_data = query.overlay_data) === null || _query$overlay_data === void 0 ? void 0 : _query$overlay_data.split('&').map(function (plot) {
        var parts = plot.split('/');
        var run_number = parts.shift();
        var pathAndLabel = parts.splice(3);
        var dataset_name = parts.join('/');
        var path = selected_plot.path;
        var plot_name = selected_plot.name;
        var label = pathAndLabel.pop();
        var string = [run_number, dataset_name, path, plot_name, label].join('/');
        return string;
      });
      var global_overlay = 'overlaidGlobally=' + globally_overlaid_plots.join('&');
      var baseURL = [basePath, page].join('/');
      var queryURL = [run, dataset, path, plot_name, global_overlay].join('&');
      var plotsLocalOverlayURL = [baseURL, queryURL].join('?');
      return plotsLocalOverlayURL;
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 92,
        columnNumber: 13
      }
    })
  }];
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;
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
      lineNumber: 113,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_11__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 114,
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
      lineNumber: 119,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 128,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 131,
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
      lineNumber: 136,
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
      lineNumber: 147,
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
      lineNumber: 153,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 162,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 165,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_9__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 166,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_8__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 167,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ImageDiv"], {
    id: selected_plot.name,
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 171,
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
      lineNumber: 176,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRPdmVybGF5UGxvdHMvem9vbWVkT3ZlcmxhaWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRPdmVybGFpZFBsb3QiLCJzZWxlY3RlZF9wbG90IiwicGFyYW1zX2Zvcl9hcGkiLCJ1c2VTdGF0ZSIsImN1c3RvbWl6YXRpb25QYXJhbXMiLCJzZXRDdXN0b21pemF0aW9uUGFyYW1zIiwib3BlbkN1c3RvbWl6YXRpb24iLCJ0b2dnbGVDdXN0b21pemF0aW9uTWVudSIsImN1c3RvbWl6ZVByb3BzIiwiUmVhY3QiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJ6b29tZWRQbG90TWVudU9wdGlvbnMiLCJsYWJlbCIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJiYXNlUGF0aCIsInJvdXRlciIsInBhZ2UiLCJydW4iLCJxdWVyeSIsInJ1bl9udW1iZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwicGF0aCIsInBsb3RfbmFtZSIsIm5hbWUiLCJnbG9iYWxseV9vdmVybGFpZF9wbG90cyIsIm92ZXJsYXlfZGF0YSIsInNwbGl0IiwibWFwIiwicGxvdCIsInBhcnRzIiwic2hpZnQiLCJwYXRoQW5kTGFiZWwiLCJzcGxpY2UiLCJqb2luIiwicG9wIiwic3RyaW5nIiwiZ2xvYmFsX292ZXJsYXkiLCJiYXNlVVJMIiwicXVlcnlVUkwiLCJwbG90c0xvY2FsT3ZlcmxheVVSTCIsInVzZVJvdXRlciIsIm92ZXJsYWlkX3Bsb3RzX3VybHMiLCJnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMiLCJqb2luZWRfb3ZlcmxhaWRfcGxvdHNfdXJscyIsImpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJscyIsInNvdXJjZSIsImdldF9wbG90X3NvdXJjZSIsImNvcHlfb2ZfcGFyYW1zIiwiaGVpZ2h0Iiwid2luZG93IiwiaW5uZXJIZWlnaHQiLCJ3aWR0aCIsIk1hdGgiLCJyb3VuZCIsInpvb21lZF9wbG90X3VybCIsInVzZUJsaW5rT25VcGRhdGUiLCJibGluayIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJ0b1N0cmluZyIsIm1vZGUiLCJnZXRfcGxvdF9lcnJvciIsInJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUVBO0FBVUE7QUFDQTtBQVNBO0FBSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQU9PLElBQU1BLGtCQUFrQixHQUFHLFNBQXJCQSxrQkFBcUIsT0FHVjtBQUFBOztBQUFBOztBQUFBLE1BRnRCQyxhQUVzQixRQUZ0QkEsYUFFc0I7QUFBQSxNQUR0QkMsY0FDc0IsUUFEdEJBLGNBQ3NCOztBQUFBLGtCQUNnQ0Msc0RBQVEsRUFEeEM7QUFBQSxNQUNmQyxtQkFEZTtBQUFBLE1BQ01DLHNCQUROOztBQUFBLG1CQUkrQkYsc0RBQVEsQ0FBQyxLQUFELENBSnZDO0FBQUEsTUFJZkcsaUJBSmU7QUFBQSxNQUlJQyx1QkFKSjs7QUFLdEJMLGdCQUFjLENBQUNNLGNBQWYsR0FBZ0NKLG1CQUFoQzs7QUFMc0Isd0JBTThCSyw0Q0FBSyxDQUFDTixRQUFOLENBQWUsS0FBZixDQU45QjtBQUFBO0FBQUEsTUFNZk8sa0JBTmU7QUFBQSxNQU1LQyxxQkFOTDs7QUFRdEIsTUFBTUMscUJBQXFCLEdBQUcsQ0FDNUI7QUFDRUMsU0FBSyxFQUFFLG1CQURUO0FBRUVDLFNBQUssRUFBRSxtQkFGVDtBQUdFQyxVQUFNLEVBQUU7QUFBQSxhQUFNSixxQkFBcUIsQ0FBQyxJQUFELENBQTNCO0FBQUEsS0FIVjtBQUlFSyxRQUFJLEVBQUUsTUFBQyxvRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBSlIsR0FENEIsRUFPNUI7QUFDRUgsU0FBSyxFQUFF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QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBTEYsQ0FsQkYsQ0F4Q0YsQ0FERjtBQTRFRCxDQW5KTTs7R0FBTS9ELGtCO1VBdURJOEMscUQsRUFjOEJlLHlFOzs7S0FyRWxDN0Qsa0IiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNzQwZWM1NTMzZDlkMTIzNjYzYjAuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBTdG9yZSB9IGZyb20gJ2FudGQvbGliL2Zvcm0vaW50ZXJmYWNlJztcclxuaW1wb3J0IHsgU2V0dGluZ091dGxpbmVkLCBGdWxsc2NyZWVuT3V0bGluZWQsIEJsb2NrT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQge1xyXG4gIGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyxcclxuICBmdW5jdGlvbnNfY29uZmlnLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFBsb3REYXRhUHJvcHMsXHJcbiAgUXVlcnlQcm9wcyxcclxuICBDdXN0b21pemVQcm9wcyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IGdldF9wbG90X3NvdXJjZSB9IGZyb20gJy4vdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZFBsb3RSb3csXHJcbiAgUGxvdE5hbWVDb2wsXHJcbiAgQ29sdW1uLFxyXG4gIFN0eWxlZENvbCxcclxuICBJbWFnZURpdixcclxuICBJbWFnZSxcclxuICBNaW51c0ljb24sXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlLFxyXG4gIGdldF9wbG90X2Vycm9yLFxyXG59IGZyb20gJy4uLy4uL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IFpvb21lZFBsb3RNZW51IH0gZnJvbSAnLi4vbWVudSc7XHJcbmltcG9ydCB7IEN1c3RvbWl6YXRpb24gfSBmcm9tICcuLi8uLi8uLi9jdXN0b21pemF0aW9uJztcclxuaW1wb3J0IHsgUGxvdF9wb3J0YWwgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvcG9ydGFsJztcclxuaW1wb3J0IHsgdXNlQmxpbmtPblVwZGF0ZSB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZUJsaW5rT25VcGRhdGUnO1xyXG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi8uLi9wbG90L3Bsb3RJbWFnZSc7XHJcblxyXG5pbnRlcmZhY2UgWm9vbWVkUGxvdHNQcm9wcyB7XHJcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcclxuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRPdmVybGFpZFBsb3QgPSAoe1xyXG4gIHNlbGVjdGVkX3Bsb3QsXHJcbiAgcGFyYW1zX2Zvcl9hcGksXHJcbn06IFpvb21lZFBsb3RzUHJvcHMpID0+IHtcclxuICBjb25zdCBbY3VzdG9taXphdGlvblBhcmFtcywgc2V0Q3VzdG9taXphdGlvblBhcmFtc10gPSB1c2VTdGF0ZTxcclxuICAgIFBhcnRpYWw8U3RvcmU+ICYgQ3VzdG9taXplUHJvcHNcclxuICA+KCk7XHJcbiAgY29uc3QgW29wZW5DdXN0b21pemF0aW9uLCB0b2dnbGVDdXN0b21pemF0aW9uTWVudV0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgcGFyYW1zX2Zvcl9hcGkuY3VzdG9taXplUHJvcHMgPSBjdXN0b21pemF0aW9uUGFyYW1zO1xyXG4gIGNvbnN0IFtpc1BvcnRhbFdpbmRvd09wZW4sIHNldElzUG9ydGFsV2luZG93T3Blbl0gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIGNvbnN0IHpvb21lZFBsb3RNZW51T3B0aW9ucyA9IFtcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdPcGVuIGluIGEgbmV3IHRhYicsXHJcbiAgICAgIHZhbHVlOiAnb3Blbl9pbl9hX25ld190YWInLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHNldElzUG9ydGFsV2luZG93T3Blbih0cnVlKSxcclxuICAgICAgaWNvbjogPEZ1bGxzY3JlZW5PdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgICB7XHJcbiAgICAgIGxhYmVsOiAnQ3VzdG9taXplJyxcclxuICAgICAgdmFsdWU6ICdDdXN0b21pemUnLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KHRydWUpLFxyXG4gICAgICBpY29uOiA8U2V0dGluZ091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCAmJiB7XHJcbiAgICAgIGxhYmVsOiAnT3ZlcmxheSB3aXRoIGFub3RoZXIgcGxvdCcsXHJcbiAgICAgIHZhbHVlOiAnb3ZlcmxheScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGJhc2VQYXRoID0gcm91dGVyLmJhc2VQYXRoXHJcbiAgICAgICAgY29uc3QgcGFnZSA9ICdwbG90c0xvY2FsT3ZlcmxheSdcclxuICAgICAgICBjb25zdCBydW4gPSAncnVuX251bWJlcj0nICsgcXVlcnkucnVuX251bWJlciBhcyBzdHJpbmdcclxuICAgICAgICBjb25zdCBkYXRhc2V0ID0gJ2RhdGFzZXRfbmFtZT0nICsgcXVlcnkuZGF0YXNldF9uYW1lIGFzIHN0cmluZ1xyXG4gICAgICAgIGNvbnN0IHBhdGggPSAnZm9sZGVyc19wYXRoPScgKyBzZWxlY3RlZF9wbG90LnBhdGhcclxuICAgICAgICBjb25zdCBwbG90X25hbWUgPSAncGxvdF9uYW1lPScgKyBzZWxlY3RlZF9wbG90Lm5hbWVcclxuICAgICAgICBjb25zdCBnbG9iYWxseV9vdmVybGFpZF9wbG90cyA9IHF1ZXJ5Lm92ZXJsYXlfZGF0YT8uc3BsaXQoJyYnKS5tYXAoKHBsb3QpID0+IHtcclxuICAgICAgICAgIGNvbnN0IHBhcnRzID0gcGxvdC5zcGxpdCgnLycpXHJcbiAgICAgICAgICBjb25zdCBydW5fbnVtYmVyID0gcGFydHMuc2hpZnQoKVxyXG4gICAgICAgICAgY29uc3QgcGF0aEFuZExhYmVsID0gcGFydHMuc3BsaWNlKDMpXHJcbiAgICAgICAgICBjb25zdCBkYXRhc2V0X25hbWUgPSBwYXJ0cy5qb2luKCcvJylcclxuICAgICAgICAgIGNvbnN0IHBhdGggPSAgc2VsZWN0ZWRfcGxvdC5wYXRoXHJcbiAgICAgICAgICBjb25zdCBwbG90X25hbWUgPSBzZWxlY3RlZF9wbG90Lm5hbWVcclxuICAgICAgICAgIGNvbnN0IGxhYmVsID0gcGF0aEFuZExhYmVsLnBvcCgpXHJcbiAgICAgICAgICBjb25zdCBzdHJpbmcgPSBbcnVuX251bWJlciwgZGF0YXNldF9uYW1lLCBwYXRoLCBwbG90X25hbWUsIGxhYmVsXS5qb2luKCcvJylcclxuICAgICAgICAgIHJldHVybiBzdHJpbmdcclxuICAgICAgICB9KVxyXG4gICAgICAgIGNvbnN0IGdsb2JhbF9vdmVybGF5ID0gJ292ZXJsYWlkR2xvYmFsbHk9JyArIChnbG9iYWxseV9vdmVybGFpZF9wbG90cyBhcyBzdHJpbmdbXSkuam9pbignJicpXHJcbiAgICAgICAgY29uc3QgYmFzZVVSTCA9IFtiYXNlUGF0aCwgcGFnZV0uam9pbignLycpXHJcbiAgICAgICAgY29uc3QgcXVlcnlVUkwgPSBbcnVuLCBkYXRhc2V0LCBwYXRoLCBwbG90X25hbWUsIGdsb2JhbF9vdmVybGF5XS5qb2luKCcmJylcclxuICAgICAgICBjb25zdCBwbG90c0xvY2FsT3ZlcmxheVVSTCA9IFtiYXNlVVJMLCBxdWVyeVVSTF0uam9pbignPycpXHJcbiAgICAgICAgcmV0dXJuIHBsb3RzTG9jYWxPdmVybGF5VVJMXHJcbiAgICAgIH0sXHJcbiAgICAgIGljb246IDxCbG9ja091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICBdO1xyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuXHJcbiAgY29uc3Qgb3ZlcmxhaWRfcGxvdHNfdXJscyA9IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyhwYXJhbXNfZm9yX2FwaSk7XHJcbiAgY29uc3Qgam9pbmVkX292ZXJsYWlkX3Bsb3RzX3VybHMgPSBvdmVybGFpZF9wbG90c191cmxzLmpvaW4oJycpO1xyXG4gIHBhcmFtc19mb3JfYXBpLmpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJscyA9IGpvaW5lZF9vdmVybGFpZF9wbG90c191cmxzO1xyXG5cclxuICBjb25zdCBzb3VyY2UgPSBnZXRfcGxvdF9zb3VyY2UocGFyYW1zX2Zvcl9hcGkpO1xyXG5cclxuICBjb25zdCBjb3B5X29mX3BhcmFtcyA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XHJcbiAgY29weV9vZl9wYXJhbXMud2lkdGggPSBNYXRoLnJvdW5kKHdpbmRvdy5pbm5lckhlaWdodCAqIDEuMzMpO1xyXG4gIGNvbnN0IHpvb21lZF9wbG90X3VybCA9IGdldF9wbG90X3NvdXJjZShjb3B5X29mX3BhcmFtcyk7XHJcblxyXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICA8UGxvdF9wb3J0YWxcclxuICAgICAgICBpc1BvcnRhbFdpbmRvd09wZW49e2lzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICBzZXRJc1BvcnRhbFdpbmRvd09wZW49e3NldElzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICB0aXRsZT17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFN0eWxlZFBsb3RSb3dcclxuICAgICAgICAgIGp1c3RpZnljb250ZW50PVwiY2VudGVyXCJcclxuICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBtaW5oZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aH1cclxuICAgICAgICAgICAgaGVpZ2h0PXtjb3B5X29mX3BhcmFtcy5oZWlnaHR9XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e2NvcHlfb2ZfcGFyYW1zfVxyXG4gICAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgICAgcGxvdFVSTD17em9vbWVkX3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgICAvPlxyXG4gICAgICAgICAgPC9JbWFnZURpdj5cclxuICAgICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICAgIDwvUGxvdF9wb3J0YWw+XHJcbiAgICAgIDxDdXN0b21pemF0aW9uXHJcbiAgICAgICAgcGxvdF9uYW1lPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgb3Blbj17b3BlbkN1c3RvbWl6YXRpb259XHJcbiAgICAgICAgb25DYW5jZWw9eygpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KGZhbHNlKX1cclxuICAgICAgICBzZXRDdXN0b21pemF0aW9uUGFyYW1zPXtzZXRDdXN0b21pemF0aW9uUGFyYW1zfVxyXG4gICAgICAvPlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxyXG4gICAgICA+XHJcbiAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICA8Q29sdW1uIGRpc3BsYXk9XCJmbGV4XCI+XHJcbiAgICAgICAgICA8Wm9vbWVkUGxvdE1lbnUgb3B0aW9ucz17em9vbWVkUGxvdE1lbnVPcHRpb25zfSAvPlxyXG4gICAgICAgICAgPE1pbnVzSWNvblxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZShxdWVyeSwgc2VsZWN0ZWRfcGxvdCl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sdW1uPlxyXG4gICAgICAgIDxJbWFnZURpdlxyXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aH1cclxuICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XHJcbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgIHBsb3RVUkw9e3NvdXJjZX1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
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
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
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



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidPlot.tsx",
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

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(true),
      imageLoading = _useState3[0],
      setImageLoading = _useState3[1];

  var _useState4 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(false),
      imageError = _useState4[0],
      setImageError = _useState4[1];

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
        lineNumber: 59,
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
        lineNumber: 65,
        columnNumber: 13
      }
    })
  } // {
  //   label: 'Overlay with another plot',
  //   value: 'Customize',
  //   action: () => toggleCustomizationMenu(true),
  //   icon: <BlockOutlined  />,
  // },
  ];
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
      lineNumber: 92,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_11__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.displayedName,
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
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ImageDiv"], {
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
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_7__["Column"], {
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

_s(ZoomedOverlaidPlot, "+ICec4fjFB5ypov5Nry/7bQgbsM=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRPdmVybGF5UGxvdHMvem9vbWVkT3ZlcmxhaWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRPdmVybGFpZFBsb3QiLCJzZWxlY3RlZF9wbG90IiwicGFyYW1zX2Zvcl9hcGkiLCJ1c2VTdGF0ZSIsImN1c3RvbWl6YXRpb25QYXJhbXMiLCJzZXRDdXN0b21pemF0aW9uUGFyYW1zIiwib3BlbkN1c3RvbWl6YXRpb24iLCJ0b2dnbGVDdXN0b21pemF0aW9uTWVudSIsImN1c3RvbWl6ZVByb3BzIiwiaW1hZ2VMb2FkaW5nIiwic2V0SW1hZ2VMb2FkaW5nIiwiaW1hZ2VFcnJvciIsInNldEltYWdlRXJyb3IiLCJSZWFjdCIsImlzUG9ydGFsV2luZG93T3BlbiIsInNldElzUG9ydGFsV2luZG93T3BlbiIsInpvb21lZFBsb3RNZW51T3B0aW9ucyIsImxhYmVsIiwidmFsdWUiLCJhY3Rpb24iLCJpY29uIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJvdmVybGFpZF9wbG90c191cmxzIiwiZ2V0X292ZXJsYWllZF9wbG90c191cmxzIiwiam9pbmVkX292ZXJsYWlkX3Bsb3RzX3VybHMiLCJqb2luIiwiam9pbmVkX292ZXJsYWllZF9wbG90c191cmxzIiwic291cmNlIiwiZ2V0X3Bsb3Rfc291cmNlIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsImRpc3BsYXllZE5hbWUiLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiZ2V0X3Bsb3RfZXJyb3IiLCJuYW1lIiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiXSwibWFwcGluZ3Mi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QSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFGRixDQVpGLEVBa0JFLE1BQUMsNkVBQUQ7QUFDRSxNQUFFLEVBQUVBLGFBQWEsQ0FBQzhDLElBRHBCO0FBRUUsU0FBSyxFQUFFN0MsY0FBYyxDQUFDaUMsS0FGeEI7QUFHRSxVQUFNLEVBQUVqQyxjQUFjLENBQUM4QixNQUh6QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0UsTUFBQywwREFBRDtBQUNFLFNBQUssRUFBRVEsS0FEVDtBQUVFLGtCQUFjLEVBQUV0QyxjQUZsQjtBQUdFLFFBQUksRUFBRUQsYUFIUjtBQUlFLFdBQU8sRUFBRTRCLE1BSlg7QUFLRSxTQUFLLEVBQUVOLEtBTFQ7QUFNRSw2QkFBeUIsRUFBRWtCLHlCQU43QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBTEYsQ0FsQkYsQ0F4Q0YsQ0FERjtBQTRFRCxDQTlITTs7R0FBTXpDLGtCO1VBa0NJc0IscUQsRUFjOEJpQix5RTs7O0tBaERsQ3ZDLGtCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjIyNzAwNmYwYzJmZTc0ZTZkNThiLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBTdG9yZSB9IGZyb20gJ2FudGQvbGliL2Zvcm0vaW50ZXJmYWNlJztcbmltcG9ydCB7IFNldHRpbmdPdXRsaW5lZCwgRnVsbHNjcmVlbk91dGxpbmVkLCBCbG9ja091dGxpbmVkICB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcblxuaW1wb3J0IHtcbiAgZ2V0X292ZXJsYWllZF9wbG90c191cmxzLFxuICBmdW5jdGlvbnNfY29uZmlnLFxufSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7XG4gIFBhcmFtc0ZvckFwaVByb3BzLFxuICBQbG90RGF0YVByb3BzLFxuICBRdWVyeVByb3BzLFxuICBDdXN0b21pemVQcm9wcyxcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgZ2V0X3Bsb3Rfc291cmNlIH0gZnJvbSAnLi91dGlscyc7XG5pbXBvcnQge1xuICBTdHlsZWRQbG90Um93LFxuICBQbG90TmFtZUNvbCxcbiAgQ29sdW1uLFxuICBTdHlsZWRDb2wsXG4gIEltYWdlRGl2LFxuICBJbWFnZSxcbiAgTWludXNJY29uLFxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQge1xuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcbiAgZ2V0X3Bsb3RfZXJyb3IsXG59IGZyb20gJy4uLy4uL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XG5pbXBvcnQgeyBab29tZWRQbG90TWVudSB9IGZyb20gJy4uL21lbnUnO1xuaW1wb3J0IHsgQ3VzdG9taXphdGlvbiB9IGZyb20gJy4uLy4uLy4uL2N1c3RvbWl6YXRpb24nO1xuaW1wb3J0IHsgUGxvdF9wb3J0YWwgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvcG9ydGFsJztcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJztcbmltcG9ydCB7IFBsb3RJbWFnZSB9IGZyb20gJy4uLy4uL3Bsb3QvcGxvdEltYWdlJztcblxuaW50ZXJmYWNlIFpvb21lZFBsb3RzUHJvcHMge1xuICBzZWxlY3RlZF9wbG90OiBQbG90RGF0YVByb3BzO1xuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XG59XG5cbmV4cG9ydCBjb25zdCBab29tZWRPdmVybGFpZFBsb3QgPSAoe1xuICBzZWxlY3RlZF9wbG90LFxuICBwYXJhbXNfZm9yX2FwaSxcbn06IFpvb21lZFBsb3RzUHJvcHMpID0+IHtcbiAgY29uc3QgW2N1c3RvbWl6YXRpb25QYXJhbXMsIHNldEN1c3RvbWl6YXRpb25QYXJhbXNdID0gdXNlU3RhdGU8XG4gICAgUGFydGlhbDxTdG9yZT4gJiBDdXN0b21pemVQcm9wc1xuICA+KCk7XG4gIGNvbnN0IFtvcGVuQ3VzdG9taXphdGlvbiwgdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnVdID0gdXNlU3RhdGUoZmFsc2UpO1xuICBwYXJhbXNfZm9yX2FwaS5jdXN0b21pemVQcm9wcyA9IGN1c3RvbWl6YXRpb25QYXJhbXM7XG4gIGNvbnN0IFtpbWFnZUxvYWRpbmcsIHNldEltYWdlTG9hZGluZ10gPSB1c2VTdGF0ZSh0cnVlKTtcbiAgY29uc3QgW2ltYWdlRXJyb3IsIHNldEltYWdlRXJyb3JdID0gdXNlU3RhdGUoZmFsc2UpO1xuICBjb25zdCBbaXNQb3J0YWxXaW5kb3dPcGVuLCBzZXRJc1BvcnRhbFdpbmRvd09wZW5dID0gUmVhY3QudXNlU3RhdGUoZmFsc2UpO1xuXG4gIGNvbnN0IHpvb21lZFBsb3RNZW51T3B0aW9ucyA9IFtcbiAgICB7XG4gICAgICBsYWJlbDogJ09wZW4gaW4gYSBuZXcgdGFiJyxcbiAgICAgIHZhbHVlOiAnb3Blbl9pbl9hX25ld190YWInLFxuICAgICAgYWN0aW9uOiAoKSA9PiBzZXRJc1BvcnRhbFdpbmRvd09wZW4odHJ1ZSksXG4gICAgICBpY29uOiA8RnVsbHNjcmVlbk91dGxpbmVkIC8+LFxuICAgIH0sXG4gICAge1xuICAgICAgbGFiZWw6ICdDdXN0b21pemUnLFxuICAgICAgdmFsdWU6ICdDdXN0b21pemUnLFxuICAgICAgYWN0aW9uOiAoKSA9PiB0b2dnbGVDdXN0b21pemF0aW9uTWVudSh0cnVlKSxcbiAgICAgIGljb246IDxTZXR0aW5nT3V0bGluZWQgLz4sXG4gICAgfSxcbiAgICAvLyB7XG4gICAgLy8gICBsYWJlbDogJ092ZXJsYXkgd2l0aCBhbm90aGVyIHBsb3QnLFxuICAgIC8vICAgdmFsdWU6ICdDdXN0b21pemUnLFxuICAgIC8vICAgYWN0aW9uOiAoKSA9PiB0b2dnbGVDdXN0b21pemF0aW9uTWVudSh0cnVlKSxcbiAgICAvLyAgIGljb246IDxCbG9ja091dGxpbmVkICAvPixcbiAgICAvLyB9LFxuICBdO1xuXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCBvdmVybGFpZF9wbG90c191cmxzID0gZ2V0X292ZXJsYWllZF9wbG90c191cmxzKHBhcmFtc19mb3JfYXBpKTtcbiAgY29uc3Qgam9pbmVkX292ZXJsYWlkX3Bsb3RzX3VybHMgPSBvdmVybGFpZF9wbG90c191cmxzLmpvaW4oJycpO1xuICBwYXJhbXNfZm9yX2FwaS5qb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMgPSBqb2luZWRfb3ZlcmxhaWRfcGxvdHNfdXJscztcblxuICBjb25zdCBzb3VyY2UgPSBnZXRfcGxvdF9zb3VyY2UocGFyYW1zX2Zvcl9hcGkpO1xuXG4gIGNvbnN0IGNvcHlfb2ZfcGFyYW1zID0geyAuLi5wYXJhbXNfZm9yX2FwaSB9O1xuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XG4gIGNvcHlfb2ZfcGFyYW1zLndpZHRoID0gTWF0aC5yb3VuZCh3aW5kb3cuaW5uZXJIZWlnaHQgKiAxLjMzKTtcbiAgY29uc3Qgem9vbWVkX3Bsb3RfdXJsID0gZ2V0X3Bsb3Rfc291cmNlKGNvcHlfb2ZfcGFyYW1zKTtcblxuICBjb25zdCB7IGJsaW5rLCB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XG5cbiAgcmV0dXJuIChcbiAgICA8U3R5bGVkQ29sIHNwYWNlPXsyfT5cbiAgICAgIDxQbG90X3BvcnRhbFxuICAgICAgICBpc1BvcnRhbFdpbmRvd09wZW49e2lzUG9ydGFsV2luZG93T3Blbn1cbiAgICAgICAgc2V0SXNQb3J0YWxXaW5kb3dPcGVuPXtzZXRJc1BvcnRhbFdpbmRvd09wZW59XG4gICAgICAgIHRpdGxlPXtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XG4gICAgICA+XG4gICAgICAgIDxTdHlsZWRQbG90Um93XG4gICAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxuICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cbiAgICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxuICAgICAgICAgIG1pbmhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxuICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aD8udG9TdHJpbmcoKX1cbiAgICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XG4gICAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XG4gICAgICAgID5cbiAgICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxuICAgICAgICAgICAge3NlbGVjdGVkX3Bsb3QuZGlzcGxheWVkTmFtZX1cbiAgICAgICAgICA8L1Bsb3ROYW1lQ29sPlxuICAgICAgICAgIDxJbWFnZURpdlxuICAgICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cbiAgICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aH1cbiAgICAgICAgICAgIGhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxuICAgICAgICAgID5cbiAgICAgICAgICAgIDxQbG90SW1hZ2VcbiAgICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxuICAgICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17Y29weV9vZl9wYXJhbXN9XG4gICAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XG4gICAgICAgICAgICAgIHBsb3RVUkw9e3pvb21lZF9wbG90X3VybH1cbiAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxuICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxuICAgICAgICAgICAgLz5cbiAgICAgICAgICA8L0ltYWdlRGl2PlxuICAgICAgICA8L1N0eWxlZFBsb3RSb3c+XG4gICAgICA8L1Bsb3RfcG9ydGFsPlxuICAgICAgPEN1c3RvbWl6YXRpb25cbiAgICAgICAgcGxvdF9uYW1lPXtzZWxlY3RlZF9wbG90Lm5hbWV9XG4gICAgICAgIG9wZW49e29wZW5DdXN0b21pemF0aW9ufVxuICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUoZmFsc2UpfVxuICAgICAgICBzZXRDdXN0b21pemF0aW9uUGFyYW1zPXtzZXRDdXN0b21pemF0aW9uUGFyYW1zfVxuICAgICAgLz5cbiAgICAgIDxTdHlsZWRQbG90Um93XG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cbiAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cbiAgICAgICAgbWluaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cbiAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxuICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxuICAgICAgPlxuICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XG4gICAgICAgIDxDb2x1bW4gZGlzcGxheT1cImZsZXhcIj5cbiAgICAgICAgICA8Wm9vbWVkUGxvdE1lbnUgb3B0aW9ucz17em9vbWVkUGxvdE1lbnVPcHRpb25zfSAvPlxuICAgICAgICAgIDxNaW51c0ljb25cbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0NvbHVtbj5cbiAgICAgICAgPEltYWdlRGl2XG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XG4gICAgICAgID5cbiAgICAgICAgICA8UGxvdEltYWdlXG4gICAgICAgICAgICBibGluaz17Ymxpbmt9XG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XG4gICAgICAgICAgICBwbG90PXtzZWxlY3RlZF9wbG90fVxuICAgICAgICAgICAgcGxvdFVSTD17c291cmNlfVxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0ltYWdlRGl2PlxuICAgICAgPC9TdHlsZWRQbG90Um93PlxuICAgIDwvU3R5bGVkQ29sPlxuICApO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=
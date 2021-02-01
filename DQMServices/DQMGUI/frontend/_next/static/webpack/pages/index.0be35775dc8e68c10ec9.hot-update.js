webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/zoomedPlots/zoomedPlots/zoomedPlot.tsx":
/*!*****************************************************************!*\
  !*** ./components/plots/zoomedPlots/zoomedPlots/zoomedPlot.tsx ***!
  \*****************************************************************/
/*! exports provided: ZoomedPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedPlot", function() { return ZoomedPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/defineProperty */ "./node_modules/@babel/runtime/helpers/esm/defineProperty.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _customization__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../../customization */ "./components/customization/index.tsx");
/* harmony import */ var _menu__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../menu */ "./components/plots/zoomedPlots/menu.tsx");
/* harmony import */ var _containers_display_portal__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../../../containers/display/portal */ "./containers/display/portal/index.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");
/* harmony import */ var _plot_plotImage__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../plot/plotImage */ "./components/plots/plot/plotImage.tsx");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../../utils */ "./components/utils.ts");


var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/zoomedPlots/zoomedPlot.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1___default.a.createElement;

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__["default"])(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }













var ZoomedPlot = function ZoomedPlot(_ref) {
  _s();

  var _copy_of_params$width, _params_for_api$width;

  var selected_plot = _ref.selected_plot,
      params_for_api = _ref.params_for_api;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(),
      customizationParams = _useState[0],
      setCustomizationParams = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(false),
      openCustomization = _useState2[0],
      toggleCustomizationMenu = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(false),
      isPortalWindowOpen = _useState3[0],
      setIsPortalWindowOpen = _useState3[1];

  params_for_api.customizeProps = customizationParams;
  var plot_url = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_url"])(params_for_api);

  var copy_of_params = _objectSpread({}, params_for_api);

  copy_of_params.height = window.innerHeight;
  copy_of_params.width = Math.round(window.innerHeight * 1.33);
  var zoomed_plot_url = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_url"])(copy_of_params);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var url = Object(_utils__WEBPACK_IMPORTED_MODULE_12__["getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames"])(router.basePath, query, selected_plot);
  var zoomedPlotMenuOptions = [{
    label: 'Open in a new tab',
    value: 'open_in_a_new_tab',
    action: function action() {
      return setIsPortalWindowOpen(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["FullscreenOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 68,
        columnNumber: 13
      }
    })
  }, {
    label: 'Customize',
    value: 'customize',
    action: function action() {
      return toggleCustomizationMenu(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["SettingOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 74,
        columnNumber: 13
      }
    })
  }, _config_config__WEBPACK_IMPORTED_MODULE_4__["functions_config"].new_back_end.new_back_end && {
    label: 'Overlay with another plot',
    value: 'overlay',
    url: url,
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 80,
        columnNumber: 13
      }
    })
  }];

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink,
      updated_by_not_older_than = _useBlinkOnUpdate.updated_by_not_older_than;

  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 86,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_9__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledPlotRow"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_4__["functions_config"].mode === 'ONLINE').toString(),
    minheight: copy_of_params.height,
    width: (_copy_of_params$width = copy_of_params.width) === null || _copy_of_params$width === void 0 ? void 0 : _copy_of_params$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 101,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 104,
      columnNumber: 11
    }
  }, __jsx(_plot_plotImage__WEBPACK_IMPORTED_MODULE_11__["PlotImage"], {
    blink: blink,
    params_for_api: copy_of_params,
    plot: selected_plot,
    plotURL: zoomed_plot_url,
    query: query,
    updated_by_not_older_than: updated_by_not_older_than,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 109,
      columnNumber: 13
    }
  })))), __jsx(_customization__WEBPACK_IMPORTED_MODULE_7__["Customization"], {
    plot_name: selected_plot.name,
    open: openCustomization,
    onCancel: function onCancel() {
      return toggleCustomizationMenu(false);
    },
    setCustomizationParams: setCustomizationParams,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 121,
      columnNumber: 7
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledPlotRow"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_4__["functions_config"].mode === 'ONLINE').toString(),
    minheight: params_for_api.height,
    width: (_params_for_api$width = params_for_api.width) === null || _params_for_api$width === void 0 ? void 0 : _params_for_api$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 127,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 135,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 138,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 139,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 140,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    alignitems: "center",
    id: selected_plot.name,
    width: params_for_api.width,
    height: params_for_api.height,
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 144,
      columnNumber: 9
    }
  }, __jsx(_plot_plotImage__WEBPACK_IMPORTED_MODULE_11__["PlotImage"], {
    updated_by_not_older_than: updated_by_not_older_than,
    blink: blink,
    params_for_api: params_for_api,
    plot: selected_plot,
    plotURL: plot_url,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 151,
      columnNumber: 11
    }
  }))));
};

_s(ZoomedPlot, "n7HfDH0SxZV5E2eKjp3X83/7eok=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"], _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__["useBlinkOnUpdate"]];
});

_c = ZoomedPlot;

var _c;

$RefreshReg$(_c, "ZoomedPlot");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJjdXN0b21pemVQcm9wcyIsInBsb3RfdXJsIiwiZ2V0X3Bsb3RfdXJsIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1cmwiLCJnZXRab29tZWRQbG90c1VybEZvck92ZXJsYXlpbmdQbG90c1dpdGhEaWZmZXJlbnROYW1lcyIsImJhc2VQYXRoIiwiem9vbWVkUGxvdE1lbnVPcHRpb25zIiwibGFiZWwiLCJ2YWx1ZSIsImFjdGlvbiIsImljb24iLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIm5hbWUiLCJ0b1N0cmluZyIsIm1vZGUiLCJnZXRfcGxvdF9lcnJvciIsInJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQVdBO0FBUUE7QUFJQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFPTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQUdG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QUQ7QUFDRSw2QkFBeUIsRUFBRXNCLHlCQUQ3QjtBQUVFLFNBQUssRUFBRUQsS0FGVDtBQUdFLGtCQUFjLEVBQUVqQyxjQUhsQjtBQUlFLFFBQUksRUFBRUQsYUFKUjtBQUtFLFdBQU8sRUFBRVUsUUFMWDtBQU1FLFNBQUssRUFBRVksS0FOVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBUEYsQ0FqQkYsQ0F6Q0YsQ0FERjtBQThFRCxDQTFITTs7R0FBTXZCLFU7VUFrQklzQixxRCxFQXdCOEJZLHlFOzs7S0ExQ2xDbEMsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4wYmUzNTc3NWRjOGU2OGMxMGVjOS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgRnVsbHNjcmVlbk91dGxpbmVkLCBTZXR0aW5nT3V0bGluZWQsIEJsb2NrT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcbmltcG9ydCB7IFN0b3JlIH0gZnJvbSAnYW50ZC9saWIvZm9ybS9pbnRlcmZhY2UnO1xyXG5cclxuaW1wb3J0IHtcclxuICBnZXRfcGxvdF91cmwsXHJcbiAgcm9vdF91cmwsXHJcbiAgZnVuY3Rpb25zX2NvbmZpZyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBQbG90RGF0YVByb3BzLFxyXG4gIFF1ZXJ5UHJvcHMsXHJcbiAgQ3VzdG9taXplUHJvcHMsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZENvbCxcclxuICBQbG90TmFtZUNvbCxcclxuICBTdHlsZWRQbG90Um93LFxyXG4gIENvbHVtbixcclxuICBJbWFnZURpdixcclxuICBNaW51c0ljb24sXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlLFxyXG4gIGdldF9wbG90X2Vycm9yLFxyXG59IGZyb20gJy4uLy4uL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IEN1c3RvbWl6YXRpb24gfSBmcm9tICcuLi8uLi8uLi9jdXN0b21pemF0aW9uJztcclxuaW1wb3J0IHsgWm9vbWVkUGxvdE1lbnUgfSBmcm9tICcuLi9tZW51JztcclxuaW1wb3J0IHsgUGxvdF9wb3J0YWwgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvcG9ydGFsJztcclxuaW1wb3J0IHsgdXNlQmxpbmtPblVwZGF0ZSB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZUJsaW5rT25VcGRhdGUnO1xyXG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi8uLi9wbG90L3Bsb3RJbWFnZSc7XHJcbmltcG9ydCB7IGdldFpvb21lZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzIH0gZnJvbSAnLi4vLi4vLi4vdXRpbHMnO1xyXG5cclxuaW50ZXJmYWNlIFpvb21lZFBsb3RzUHJvcHMge1xyXG4gIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHM7XHJcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgWm9vbWVkUGxvdCA9ICh7XHJcbiAgc2VsZWN0ZWRfcGxvdCxcclxuICBwYXJhbXNfZm9yX2FwaSxcclxufTogWm9vbWVkUGxvdHNQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtjdXN0b21pemF0aW9uUGFyYW1zLCBzZXRDdXN0b21pemF0aW9uUGFyYW1zXSA9IHVzZVN0YXRlPFxyXG4gICAgUGFydGlhbDxTdG9yZT4gJiBDdXN0b21pemVQcm9wc1xyXG4gID4oKTtcclxuICBjb25zdCBbb3BlbkN1c3RvbWl6YXRpb24sIHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBbaXNQb3J0YWxXaW5kb3dPcGVuLCBzZXRJc1BvcnRhbFdpbmRvd09wZW5dID0gdXNlU3RhdGUoZmFsc2UpO1xyXG5cclxuICBwYXJhbXNfZm9yX2FwaS5jdXN0b21pemVQcm9wcyA9IGN1c3RvbWl6YXRpb25QYXJhbXM7XHJcbiAgY29uc3QgcGxvdF91cmwgPSBnZXRfcGxvdF91cmwocGFyYW1zX2Zvcl9hcGkpO1xyXG4gIGNvbnN0IGNvcHlfb2ZfcGFyYW1zID0geyAuLi5wYXJhbXNfZm9yX2FwaSB9O1xyXG4gIGNvcHlfb2ZfcGFyYW1zLmhlaWdodCA9IHdpbmRvdy5pbm5lckhlaWdodDtcclxuICBjb3B5X29mX3BhcmFtcy53aWR0aCA9IE1hdGgucm91bmQod2luZG93LmlubmVySGVpZ2h0ICogMS4zMyk7XHJcblxyXG4gIGNvbnN0IHpvb21lZF9wbG90X3VybCA9IGdldF9wbG90X3VybChjb3B5X29mX3BhcmFtcyk7XHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG5cclxuY29uc3QgdXJsID0gZ2V0Wm9vbWVkUGxvdHNVcmxGb3JPdmVybGF5aW5nUGxvdHNXaXRoRGlmZmVyZW50TmFtZXMocm91dGVyLmJhc2VQYXRoLCBxdWVyeSwgc2VsZWN0ZWRfcGxvdClcclxuICBjb25zdCB6b29tZWRQbG90TWVudU9wdGlvbnMgPSBbXHJcbiAgICB7XHJcbiAgICAgIGxhYmVsOiAnT3BlbiBpbiBhIG5ldyB0YWInLFxyXG4gICAgICB2YWx1ZTogJ29wZW5faW5fYV9uZXdfdGFiJyxcclxuICAgICAgYWN0aW9uOiAoKSA9PiBzZXRJc1BvcnRhbFdpbmRvd09wZW4odHJ1ZSksXHJcbiAgICAgIGljb246IDxGdWxsc2NyZWVuT3V0bGluZWQgLz4sXHJcbiAgICB9LFxyXG4gICAge1xyXG4gICAgICBsYWJlbDogJ0N1c3RvbWl6ZScsXHJcbiAgICAgIHZhbHVlOiAnY3VzdG9taXplJyxcclxuICAgICAgYWN0aW9uOiAoKSA9PiB0b2dnbGVDdXN0b21pemF0aW9uTWVudSh0cnVlKSxcclxuICAgICAgaWNvbjogPFNldHRpbmdPdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgICBmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5uZXdfYmFja19lbmQgJiYge1xyXG4gICAgICBsYWJlbDogJ092ZXJsYXkgd2l0aCBhbm90aGVyIHBsb3QnLFxyXG4gICAgICB2YWx1ZTogJ292ZXJsYXknLFxyXG4gICAgICB1cmw6IHVybCxcclxuICAgICAgaWNvbjogPEJsb2NrT3V0bGluZWQgLz4sXHJcbiAgICB9LFxyXG4gIF07XHJcbiAgY29uc3QgeyBibGluaywgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZENvbCBzcGFjZT17Mn0+XHJcbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XHJcbiAgICAgIDxQbG90X3BvcnRhbFxyXG4gICAgICAgIGlzUG9ydGFsV2luZG93T3Blbj17aXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHNldElzUG9ydGFsV2luZG93T3Blbj17c2V0SXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHRpdGxlPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgID5cclxuICAgICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG1pbmhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxyXG4gICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgICAge3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRofVxyXG4gICAgICAgICAgICBoZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAgPFBsb3RJbWFnZVxyXG4gICAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17Y29weV9vZl9wYXJhbXN9XHJcbiAgICAgICAgICAgICAgcGxvdD17c2VsZWN0ZWRfcGxvdH1cclxuICAgICAgICAgICAgICBwbG90VVJMPXt6b29tZWRfcGxvdF91cmx9XHJcbiAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgICAgPC9QbG90X3BvcnRhbD5cclxuICAgICAgey8qIFBsb3Qgb3BlbmVkIGluIGEgbmV3IHRhYiAqL31cclxuICAgICAgPEN1c3RvbWl6YXRpb25cclxuICAgICAgICBwbG90X25hbWU9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICBvcGVuPXtvcGVuQ3VzdG9taXphdGlvbn1cclxuICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUoZmFsc2UpfVxyXG4gICAgICAgIHNldEN1c3RvbWl6YXRpb25QYXJhbXM9e3NldEN1c3RvbWl6YXRpb25QYXJhbXN9XHJcbiAgICAgIC8+XHJcbiAgICAgIDxTdHlsZWRQbG90Um93XHJcbiAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbWluaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgPlxyXG4gICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICB7c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgPENvbHVtbiBkaXNwbGF5PVwiZmxleFwiPlxyXG4gICAgICAgICAgPFpvb21lZFBsb3RNZW51IG9wdGlvbnM9e3pvb21lZFBsb3RNZW51T3B0aW9uc30gLz5cclxuICAgICAgICAgIDxNaW51c0ljb25cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4gcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUocXVlcnksIHNlbGVjdGVkX3Bsb3QpfVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbHVtbj5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIGFsaWduaXRlbXM9XCJjZW50ZXJcIlxyXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aH1cclxuICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgICAgZGlzcGxheT1cImZsZXhcIlxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cclxuICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XHJcbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgIHBsb3RVUkw9e3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvSW1hZ2VEaXY+XHJcbiAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgIDwvU3R5bGVkQ29sPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
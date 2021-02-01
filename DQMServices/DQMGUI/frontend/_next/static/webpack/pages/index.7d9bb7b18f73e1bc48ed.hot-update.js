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
        lineNumber: 67,
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
        lineNumber: 73,
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
        lineNumber: 79,
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
      lineNumber: 85,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_9__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 87,
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
      lineNumber: 92,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 103,
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
      lineNumber: 108,
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
      lineNumber: 120,
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
      lineNumber: 126,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 134,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 137,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 138,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 139,
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
      lineNumber: 143,
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
      lineNumber: 150,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJjdXN0b21pemVQcm9wcyIsInBsb3RfdXJsIiwiZ2V0X3Bsb3RfdXJsIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1cmwiLCJnZXRab29tZWRQbG90c1VybEZvck92ZXJsYXlpbmdQbG90c1dpdGhEaWZmZXJlbnROYW1lcyIsImJhc2VQYXRoIiwiem9vbWVkUGxvdE1lbnVPcHRpb25zIiwibGFiZWwiLCJ2YWx1ZSIsImFjdGlvbiIsImljb24iLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIm5hbWUiLCJ0b1N0cmluZyIsIm1vZGUiLCJnZXRfcGxvdF9lcnJvciIsInJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQVVBO0FBUUE7QUFJQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFPTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQUdG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QUQ7QUFDRSw2QkFBeUIsRUFBRXNCLHlCQUQ3QjtBQUVFLFNBQUssRUFBRUQsS0FGVDtBQUdFLGtCQUFjLEVBQUVqQyxjQUhsQjtBQUlFLFFBQUksRUFBRUQsYUFKUjtBQUtFLFdBQU8sRUFBRVUsUUFMWDtBQU1FLFNBQUssRUFBRVksS0FOVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBUEYsQ0FqQkYsQ0F6Q0YsQ0FERjtBQThFRCxDQTFITTs7R0FBTXZCLFU7VUFrQklzQixxRCxFQXdCOEJZLHlFOzs7S0ExQ2xDbEMsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC43ZDliYjdiMThmNzNlMWJjNDhlZC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcbmltcG9ydCB7IEZ1bGxzY3JlZW5PdXRsaW5lZCwgU2V0dGluZ091dGxpbmVkLCBCbG9ja091dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5pbXBvcnQgeyBTdG9yZSB9IGZyb20gJ2FudGQvbGliL2Zvcm0vaW50ZXJmYWNlJztcclxuXHJcbmltcG9ydCB7XHJcbiAgZ2V0X3Bsb3RfdXJsLFxyXG4gIGZ1bmN0aW9uc19jb25maWcsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcbmltcG9ydCB7XHJcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXHJcbiAgUGxvdERhdGFQcm9wcyxcclxuICBRdWVyeVByb3BzLFxyXG4gIEN1c3RvbWl6ZVByb3BzLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRDb2wsXHJcbiAgUGxvdE5hbWVDb2wsXHJcbiAgU3R5bGVkUGxvdFJvdyxcclxuICBDb2x1bW4sXHJcbiAgSW1hZ2VEaXYsXHJcbiAgTWludXNJY29uLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcclxuICBnZXRfcGxvdF9lcnJvcixcclxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBDdXN0b21pemF0aW9uIH0gZnJvbSAnLi4vLi4vLi4vY3VzdG9taXphdGlvbic7XHJcbmltcG9ydCB7IFpvb21lZFBsb3RNZW51IH0gZnJvbSAnLi4vbWVudSc7XHJcbmltcG9ydCB7IFBsb3RfcG9ydGFsIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3BvcnRhbCc7XHJcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJztcclxuaW1wb3J0IHsgUGxvdEltYWdlIH0gZnJvbSAnLi4vLi4vcGxvdC9wbG90SW1hZ2UnO1xyXG5pbXBvcnQgeyBnZXRab29tZWRQbG90c1VybEZvck92ZXJsYXlpbmdQbG90c1dpdGhEaWZmZXJlbnROYW1lcyB9IGZyb20gJy4uLy4uLy4uL3V0aWxzJztcclxuXHJcbmludGVyZmFjZSBab29tZWRQbG90c1Byb3BzIHtcclxuICBzZWxlY3RlZF9wbG90OiBQbG90RGF0YVByb3BzO1xyXG4gIHBhcmFtc19mb3JfYXBpOiBQYXJhbXNGb3JBcGlQcm9wcztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3QgPSAoe1xyXG4gIHNlbGVjdGVkX3Bsb3QsXHJcbiAgcGFyYW1zX2Zvcl9hcGksXHJcbn06IFpvb21lZFBsb3RzUHJvcHMpID0+IHtcclxuICBjb25zdCBbY3VzdG9taXphdGlvblBhcmFtcywgc2V0Q3VzdG9taXphdGlvblBhcmFtc10gPSB1c2VTdGF0ZTxcclxuICAgIFBhcnRpYWw8U3RvcmU+ICYgQ3VzdG9taXplUHJvcHNcclxuICA+KCk7XHJcbiAgY29uc3QgW29wZW5DdXN0b21pemF0aW9uLCB0b2dnbGVDdXN0b21pemF0aW9uTWVudV0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3QgW2lzUG9ydGFsV2luZG93T3Blbiwgc2V0SXNQb3J0YWxXaW5kb3dPcGVuXSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuXHJcbiAgcGFyYW1zX2Zvcl9hcGkuY3VzdG9taXplUHJvcHMgPSBjdXN0b21pemF0aW9uUGFyYW1zO1xyXG4gIGNvbnN0IHBsb3RfdXJsID0gZ2V0X3Bsb3RfdXJsKHBhcmFtc19mb3JfYXBpKTtcclxuICBjb25zdCBjb3B5X29mX3BhcmFtcyA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XHJcbiAgY29weV9vZl9wYXJhbXMud2lkdGggPSBNYXRoLnJvdW5kKHdpbmRvdy5pbm5lckhlaWdodCAqIDEuMzMpO1xyXG5cclxuICBjb25zdCB6b29tZWRfcGxvdF91cmwgPSBnZXRfcGxvdF91cmwoY29weV9vZl9wYXJhbXMpO1xyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuXHJcbmNvbnN0IHVybCA9IGdldFpvb21lZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzKHJvdXRlci5iYXNlUGF0aCwgcXVlcnksIHNlbGVjdGVkX3Bsb3QpXHJcbiAgY29uc3Qgem9vbWVkUGxvdE1lbnVPcHRpb25zID0gW1xyXG4gICAge1xyXG4gICAgICBsYWJlbDogJ09wZW4gaW4gYSBuZXcgdGFiJyxcclxuICAgICAgdmFsdWU6ICdvcGVuX2luX2FfbmV3X3RhYicsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0SXNQb3J0YWxXaW5kb3dPcGVuKHRydWUpLFxyXG4gICAgICBpY29uOiA8RnVsbHNjcmVlbk91dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdDdXN0b21pemUnLFxyXG4gICAgICB2YWx1ZTogJ2N1c3RvbWl6ZScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUodHJ1ZSksXHJcbiAgICAgIGljb246IDxTZXR0aW5nT3V0bGluZWQgLz4sXHJcbiAgICB9LFxyXG4gICAgZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kICYmIHtcclxuICAgICAgbGFiZWw6ICdPdmVybGF5IHdpdGggYW5vdGhlciBwbG90JyxcclxuICAgICAgdmFsdWU6ICdvdmVybGF5JyxcclxuICAgICAgdXJsOiB1cmwsXHJcbiAgICAgIGljb246IDxCbG9ja091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICBdO1xyXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICB7LyogUGxvdCBvcGVuZWQgaW4gYSBuZXcgdGFiICovfVxyXG4gICAgICA8UGxvdF9wb3J0YWxcclxuICAgICAgICBpc1BvcnRhbFdpbmRvd09wZW49e2lzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICBzZXRJc1BvcnRhbFdpbmRvd09wZW49e3NldElzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICB0aXRsZT17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFN0eWxlZFBsb3RSb3dcclxuICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBtaW5oZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aH1cclxuICAgICAgICAgICAgaGVpZ2h0PXtjb3B5X29mX3BhcmFtcy5oZWlnaHR9XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e2NvcHlfb2ZfcGFyYW1zfVxyXG4gICAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgICAgcGxvdFVSTD17em9vbWVkX3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgICAvPlxyXG4gICAgICAgICAgPC9JbWFnZURpdj5cclxuICAgICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICAgIDwvUGxvdF9wb3J0YWw+XHJcbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XHJcbiAgICAgIDxDdXN0b21pemF0aW9uXHJcbiAgICAgICAgcGxvdF9uYW1lPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgb3Blbj17b3BlbkN1c3RvbWl6YXRpb259XHJcbiAgICAgICAgb25DYW5jZWw9eygpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KGZhbHNlKX1cclxuICAgICAgICBzZXRDdXN0b21pemF0aW9uUGFyYW1zPXtzZXRDdXN0b21pemF0aW9uUGFyYW1zfVxyXG4gICAgICAvPlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgID5cclxuICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxyXG4gICAgICAgICAge3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgIDxDb2x1bW4gZGlzcGxheT1cImZsZXhcIj5cclxuICAgICAgICAgIDxab29tZWRQbG90TWVudSBvcHRpb25zPXt6b29tZWRQbG90TWVudU9wdGlvbnN9IC8+XHJcbiAgICAgICAgICA8TWludXNJY29uXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Db2x1bW4+XHJcbiAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcclxuICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XHJcbiAgICAgICAgICBoZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICAgIGRpc3BsYXk9XCJmbGV4XCJcclxuICAgICAgICA+XHJcbiAgICAgICAgICA8UGxvdEltYWdlXHJcbiAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e3BhcmFtc19mb3JfYXBpfVxyXG4gICAgICAgICAgICBwbG90PXtzZWxlY3RlZF9wbG90fVxyXG4gICAgICAgICAgICBwbG90VVJMPXtwbG90X3VybH1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
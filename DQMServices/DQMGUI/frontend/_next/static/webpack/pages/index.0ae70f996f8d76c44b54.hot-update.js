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
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
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
/* harmony import */ var _overlayWithAnotherPlot__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../../overlayWithAnotherPlot */ "./components/overlayWithAnotherPlot/index.tsx");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/zoomedPlots/zoomedPlots/zoomedPlot.tsx",
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

  var _useState4 = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(false),
      openOverlayPlotMenu = _useState4[0],
      setOpenOverlayPlotMenu = _useState4[1];

  params_for_api.customizeProps = customizationParams;
  var plot_url = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_url"])(params_for_api);

  var copy_of_params = _objectSpread({}, params_for_api);

  copy_of_params.height = window.innerHeight;
  copy_of_params.width = Math.round(window.innerHeight * 1.33);
  var zoomed_plot_url = Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_plot_url"])(copy_of_params);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
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
        lineNumber: 69,
        columnNumber: 13
      }
    })
  }, {
    label: 'Customize',
    value: 'Customize',
    action: function action() {
      return toggleCustomizationMenu(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["SettingOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 75,
        columnNumber: 13
      }
    })
  }, {
    label: 'Overlay with another plot',
    value: 'Customize',
    action: function action() {
      return setOpenOverlayPlotMenu(true);
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 81,
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
      lineNumber: 88,
      columnNumber: 5
    }
  }, __jsx(_overlayWithAnotherPlot__WEBPACK_IMPORTED_MODULE_12__["OverlayWithAnotherPlot"], {
    visible: openOverlayPlotMenu,
    setOpenOverlayWithAnotherPlotModal: setOpenOverlayPlotMenu,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 89,
      columnNumber: 7
    }
  }), __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_9__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.displayedName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 94,
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
      lineNumber: 99,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 107,
      columnNumber: 11
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 110,
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
      lineNumber: 115,
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
      lineNumber: 127,
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
      lineNumber: 133,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 141,
      columnNumber: 9
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 144,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 145,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 146,
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
      lineNumber: 150,
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
      lineNumber: 157,
      columnNumber: 11
    }
  }))));
};

_s(ZoomedPlot, "lmhcmYuEcNOR8d66FE2VhRr5/VY=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJvcGVuT3ZlcmxheVBsb3RNZW51Iiwic2V0T3Blbk92ZXJsYXlQbG90TWVudSIsImN1c3RvbWl6ZVByb3BzIiwicGxvdF91cmwiLCJnZXRfcGxvdF91cmwiLCJjb3B5X29mX3BhcmFtcyIsImhlaWdodCIsIndpbmRvdyIsImlubmVySGVpZ2h0Iiwid2lkdGgiLCJNYXRoIiwicm91bmQiLCJ6b29tZWRfcGxvdF91cmwiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInpvb21lZFBsb3RNZW51T3B0aW9ucyIsImxhYmVsIiwidmFsdWUiLCJhY3Rpb24iLCJpY29uIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsImRpc3BsYXllZE5hbWUiLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiZ2V0X3Bsb3RfZXJyb3IiLCJuYW1lIiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBV0E7QUFRQTtBQUlBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQU9PLElBQU1BLFVBQVUsR0FBRyxTQUFiQSxVQUFhLE9BR0Y7QUFBQTs7QUFBQTs7QUFBQSxNQUZ0QkMs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RUFBQyxNQUxWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FPRSxNQUFDLDBEQUFEO0FBQ0UsNkJBQXlCLEVBQUVpQix5QkFEN0I7QUFFRSxTQUFLLEVBQUVELEtBRlQ7QUFHRSxrQkFBYyxFQUFFOUIsY0FIbEI7QUFJRSxRQUFJLEVBQUVELGFBSlI7QUFLRSxXQUFPLEVBQUVZLFFBTFg7QUFNRSxTQUFLLEVBQUVZLEtBTlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQVBGLENBakJGLENBN0NGLENBREY7QUFrRkQsQ0FoSU07O0dBQU16QixVO1VBb0JJd0IscUQsRUF3QjhCTyx5RTs7O0tBNUNsQy9CLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMGFlNzBmOTk2ZjhkNzZjNDRiNTQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0IH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgRnVsbHNjcmVlbk91dGxpbmVkLCBTZXR0aW5nT3V0bGluZWQsIEJsb2NrT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XG5pbXBvcnQgeyBTdG9yZSB9IGZyb20gJ2FudGQvbGliL2Zvcm0vaW50ZXJmYWNlJztcblxuaW1wb3J0IHtcbiAgZ2V0X3Bsb3RfdXJsLFxuICByb290X3VybCxcbiAgZnVuY3Rpb25zX2NvbmZpZyxcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQge1xuICBQYXJhbXNGb3JBcGlQcm9wcyxcbiAgUGxvdERhdGFQcm9wcyxcbiAgUXVlcnlQcm9wcyxcbiAgQ3VzdG9taXplUHJvcHMsXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7XG4gIFN0eWxlZENvbCxcbiAgUGxvdE5hbWVDb2wsXG4gIFN0eWxlZFBsb3RSb3csXG4gIENvbHVtbixcbiAgSW1hZ2VEaXYsXG4gIE1pbnVzSWNvbixcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHtcbiAgcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUsXG4gIGdldF9wbG90X2Vycm9yLFxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xuaW1wb3J0IHsgQ3VzdG9taXphdGlvbiB9IGZyb20gJy4uLy4uLy4uL2N1c3RvbWl6YXRpb24nO1xuaW1wb3J0IHsgWm9vbWVkUGxvdE1lbnUgfSBmcm9tICcuLi9tZW51JztcbmltcG9ydCB7IFBsb3RfcG9ydGFsIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3BvcnRhbCc7XG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi8uLi9wbG90L3Bsb3RJbWFnZSc7XG5pbXBvcnQgeyBPdmVybGF5V2l0aEFub3RoZXJQbG90IH0gZnJvbSAnLi4vLi4vLi4vb3ZlcmxheVdpdGhBbm90aGVyUGxvdCc7XG5cbmludGVyZmFjZSBab29tZWRQbG90c1Byb3BzIHtcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xufVxuXG5leHBvcnQgY29uc3QgWm9vbWVkUGxvdCA9ICh7XG4gIHNlbGVjdGVkX3Bsb3QsXG4gIHBhcmFtc19mb3JfYXBpLFxufTogWm9vbWVkUGxvdHNQcm9wcykgPT4ge1xuICBjb25zdCBbY3VzdG9taXphdGlvblBhcmFtcywgc2V0Q3VzdG9taXphdGlvblBhcmFtc10gPSB1c2VTdGF0ZTxcbiAgICBQYXJ0aWFsPFN0b3JlPiAmIEN1c3RvbWl6ZVByb3BzXG4gID4oKTtcbiAgY29uc3QgW29wZW5DdXN0b21pemF0aW9uLCB0b2dnbGVDdXN0b21pemF0aW9uTWVudV0gPSB1c2VTdGF0ZShmYWxzZSk7XG4gIGNvbnN0IFtpc1BvcnRhbFdpbmRvd09wZW4sIHNldElzUG9ydGFsV2luZG93T3Blbl0gPSB1c2VTdGF0ZShmYWxzZSk7XG4gIGNvbnN0IFtvcGVuT3ZlcmxheVBsb3RNZW51LCBzZXRPcGVuT3ZlcmxheVBsb3RNZW51XSA9IHVzZVN0YXRlKGZhbHNlKVxuXG4gIHBhcmFtc19mb3JfYXBpLmN1c3RvbWl6ZVByb3BzID0gY3VzdG9taXphdGlvblBhcmFtcztcbiAgY29uc3QgcGxvdF91cmwgPSBnZXRfcGxvdF91cmwocGFyYW1zX2Zvcl9hcGkpO1xuXG4gIGNvbnN0IGNvcHlfb2ZfcGFyYW1zID0geyAuLi5wYXJhbXNfZm9yX2FwaSB9O1xuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XG4gIGNvcHlfb2ZfcGFyYW1zLndpZHRoID0gTWF0aC5yb3VuZCh3aW5kb3cuaW5uZXJIZWlnaHQgKiAxLjMzKTtcblxuICBjb25zdCB6b29tZWRfcGxvdF91cmwgPSBnZXRfcGxvdF91cmwoY29weV9vZl9wYXJhbXMpO1xuXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCB6b29tZWRQbG90TWVudU9wdGlvbnMgPSBbXG4gICAge1xuICAgICAgbGFiZWw6ICdPcGVuIGluIGEgbmV3IHRhYicsXG4gICAgICB2YWx1ZTogJ29wZW5faW5fYV9uZXdfdGFiJyxcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0SXNQb3J0YWxXaW5kb3dPcGVuKHRydWUpLFxuICAgICAgaWNvbjogPEZ1bGxzY3JlZW5PdXRsaW5lZCAvPixcbiAgICB9LFxuICAgIHtcbiAgICAgIGxhYmVsOiAnQ3VzdG9taXplJyxcbiAgICAgIHZhbHVlOiAnQ3VzdG9taXplJyxcbiAgICAgIGFjdGlvbjogKCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUodHJ1ZSksXG4gICAgICBpY29uOiA8U2V0dGluZ091dGxpbmVkIC8+LFxuICAgIH0sXG4gICAge1xuICAgICAgbGFiZWw6ICdPdmVybGF5IHdpdGggYW5vdGhlciBwbG90JyxcbiAgICAgIHZhbHVlOiAnQ3VzdG9taXplJyxcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0T3Blbk92ZXJsYXlQbG90TWVudSh0cnVlKSxcbiAgICAgIGljb246IDxCbG9ja091dGxpbmVkICAvPixcbiAgICB9LFxuICBdO1xuXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcblxuICByZXR1cm4gKFxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxuICAgICAgPE92ZXJsYXlXaXRoQW5vdGhlclBsb3RcbiAgICAgICAgdmlzaWJsZT17b3Blbk92ZXJsYXlQbG90TWVudX1cbiAgICAgICAgc2V0T3Blbk92ZXJsYXlXaXRoQW5vdGhlclBsb3RNb2RhbD17c2V0T3Blbk92ZXJsYXlQbG90TWVudX1cbiAgICAgIC8+XG4gICAgICB7LyogUGxvdCBvcGVuZWQgaW4gYSBuZXcgdGFiICovfVxuICAgICAgPFBsb3RfcG9ydGFsXG4gICAgICAgIGlzUG9ydGFsV2luZG93T3Blbj17aXNQb3J0YWxXaW5kb3dPcGVufVxuICAgICAgICBzZXRJc1BvcnRhbFdpbmRvd09wZW49e3NldElzUG9ydGFsV2luZG93T3Blbn1cbiAgICAgICAgdGl0bGU9e3NlbGVjdGVkX3Bsb3QuZGlzcGxheWVkTmFtZX1cbiAgICAgID5cbiAgICAgICAgPFN0eWxlZFBsb3RSb3dcbiAgICAgICAgICBpc0xvYWRpbmc9e2JsaW5rLnRvU3RyaW5nKCl9XG4gICAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cbiAgICAgICAgICBtaW5oZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cbiAgICAgICAgICB3aWR0aD17Y29weV9vZl9wYXJhbXMud2lkdGg/LnRvU3RyaW5nKCl9XG4gICAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxuICAgICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxuICAgICAgICA+XG4gICAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cbiAgICAgICAgICAgIHtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XG4gICAgICAgICAgPC9QbG90TmFtZUNvbD5cbiAgICAgICAgICA8SW1hZ2VEaXZcbiAgICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XG4gICAgICAgICAgICB3aWR0aD17Y29weV9vZl9wYXJhbXMud2lkdGh9XG4gICAgICAgICAgICBoZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cbiAgICAgICAgICA+XG4gICAgICAgICAgICA8UGxvdEltYWdlXG4gICAgICAgICAgICAgIGJsaW5rPXtibGlua31cbiAgICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e2NvcHlfb2ZfcGFyYW1zfVxuICAgICAgICAgICAgICBwbG90PXtzZWxlY3RlZF9wbG90fVxuICAgICAgICAgICAgICBwbG90VVJMPXt6b29tZWRfcGxvdF91cmx9XG4gICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cbiAgICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgPC9JbWFnZURpdj5cbiAgICAgICAgPC9TdHlsZWRQbG90Um93PlxuICAgICAgPC9QbG90X3BvcnRhbD5cbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XG4gICAgICA8Q3VzdG9taXphdGlvblxuICAgICAgICBwbG90X25hbWU9e3NlbGVjdGVkX3Bsb3QubmFtZX1cbiAgICAgICAgb3Blbj17b3BlbkN1c3RvbWl6YXRpb259XG4gICAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVDdXN0b21pemF0aW9uTWVudShmYWxzZSl9XG4gICAgICAgIHNldEN1c3RvbWl6YXRpb25QYXJhbXM9e3NldEN1c3RvbWl6YXRpb25QYXJhbXN9XG4gICAgICAvPlxuICAgICAgPFN0eWxlZFBsb3RSb3dcbiAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxuICAgICAgICBtaW5oZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cbiAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRoPy50b1N0cmluZygpfVxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XG4gICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxuICAgICAgPlxuICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XG4gICAgICAgIDxDb2x1bW4gZGlzcGxheT1cImZsZXhcIj5cbiAgICAgICAgICA8Wm9vbWVkUGxvdE1lbnUgb3B0aW9ucz17em9vbWVkUGxvdE1lbnVPcHRpb25zfSAvPlxuICAgICAgICAgIDxNaW51c0ljb25cbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0NvbHVtbj5cbiAgICAgICAgPEltYWdlRGl2XG4gICAgICAgICAgYWxpZ25pdGVtcz1cImNlbnRlclwiXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XG4gICAgICAgICAgZGlzcGxheT1cImZsZXhcIlxuICAgICAgICA+XG4gICAgICAgICAgPFBsb3RJbWFnZVxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cbiAgICAgICAgICAgIGJsaW5rPXtibGlua31cbiAgICAgICAgICAgIHBhcmFtc19mb3JfYXBpPXtwYXJhbXNfZm9yX2FwaX1cbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XG4gICAgICAgICAgICBwbG90VVJMPXtwbG90X3VybH1cbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0ltYWdlRGl2PlxuICAgICAgPC9TdHlsZWRQbG90Um93PlxuICAgIDwvU3R5bGVkQ29sPlxuICApO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=
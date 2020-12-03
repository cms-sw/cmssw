webpackHotUpdate_N_E("pages/index",{

/***/ "./components/overlayWithAnotherPlot/index.tsx":
false,

/***/ "./components/overlayWithAnotherPlot/selectedPlotsTable.tsx":
false,

/***/ "./components/overlayWithAnotherPlot/styledComponents.ts":
false,

/***/ "./components/overlayWithAnotherPlot/utils.ts":
false,

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


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/zoomedPlots/zoomedPlots/zoomedPlot.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1___default.a.createElement;

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__["default"])(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }











 // import { OverlayWithAnotherPlot } from '../../../overlayWithAnotherPlot';

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
  }, __jsx(OverlayWithAnotherPlot, {
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

/***/ }),

/***/ "./node_modules/@ant-design/css-animation/lib/Event.js":
false,

/***/ "./node_modules/antd/lib/_util/hooks/usePatchElement.js":
false,

/***/ "./node_modules/antd/lib/_util/raf.js":
false,

/***/ "./node_modules/antd/lib/_util/unreachableException.js":
false,

/***/ "./node_modules/antd/lib/_util/wave.js":
false,

/***/ "./node_modules/antd/lib/button/LoadingIcon.js":
false,

/***/ "./node_modules/antd/lib/button/button-group.js":
false,

/***/ "./node_modules/antd/lib/button/button.js":
false,

/***/ "./node_modules/antd/lib/button/index.js":
false,

/***/ "./node_modules/antd/lib/modal/ActionButton.js":
false,

/***/ "./node_modules/antd/lib/modal/ConfirmDialog.js":
false,

/***/ "./node_modules/antd/lib/modal/Modal.js":
false,

/***/ "./node_modules/antd/lib/modal/confirm.js":
false,

/***/ "./node_modules/antd/lib/modal/useModal/HookModal.js":
false,

/***/ "./node_modules/antd/lib/modal/useModal/index.js":
false,

/***/ "./node_modules/rc-util/lib/Dom/addEventListener.js":
false

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJvcGVuT3ZlcmxheVBsb3RNZW51Iiwic2V0T3Blbk92ZXJsYXlQbG90TWVudSIsImN1c3RvbWl6ZVByb3BzIiwicGxvdF91cmwiLCJnZXRfcGxvdF91cmwiLCJjb3B5X29mX3BhcmFtcyIsImhlaWdodCIsIndpbmRvdyIsImlubmVySGVpZ2h0Iiwid2lkdGgiLCJNYXRoIiwicm91bmQiLCJ6b29tZWRfcGxvdF91cmwiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInpvb21lZFBsb3RNZW51T3B0aW9ucyIsImxhYmVsIiwidmFsdWUiLCJhY3Rpb24iLCJpY29uIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsImRpc3BsYXllZE5hbWUiLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiZ2V0X3Bsb3RfZXJyb3IiLCJuYW1lIiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7QUFXQTtBQVFBO0FBSUE7QUFDQTtBQUNBO0FBQ0E7Q0FFQTs7QUFPTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQUdGO0FBQUE7O0FBQUE7O0FBQUEs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RSxXQUFPLEVBQUMsTUFMVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBT0UsTUFBQywwREFBRDtBQUNFLDZCQUF5QixFQUFFaUIseUJBRDdCO0FBRUUsU0FBSyxFQUFFRCxLQUZUO0FBR0Usa0JBQWMsRUFBRTlCLGNBSGxCO0FBSUUsUUFBSSxFQUFFRCxhQUpSO0FBS0UsV0FBTyxFQUFFWSxRQUxYO0FBTUUsU0FBSyxFQUFFWSxLQU5UO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFQRixDQWpCRixDQTdDRixDQURGO0FBa0ZELENBaElNOztHQUFNekIsVTtVQW9CSXdCLHFELEVBd0I4Qk8seUU7OztLQTVDbEMvQixVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjAyMTExNjk0Y2U3ZTUyYzE5Mzg2LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGdWxsc2NyZWVuT3V0bGluZWQsIFNldHRpbmdPdXRsaW5lZCwgQmxvY2tPdXRsaW5lZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuaW1wb3J0IHsgU3RvcmUgfSBmcm9tICdhbnRkL2xpYi9mb3JtL2ludGVyZmFjZSc7XHJcblxyXG5pbXBvcnQge1xyXG4gIGdldF9wbG90X3VybCxcclxuICByb290X3VybCxcclxuICBmdW5jdGlvbnNfY29uZmlnLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFBsb3REYXRhUHJvcHMsXHJcbiAgUXVlcnlQcm9wcyxcclxuICBDdXN0b21pemVQcm9wcyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkQ29sLFxyXG4gIFBsb3ROYW1lQ29sLFxyXG4gIFN0eWxlZFBsb3RSb3csXHJcbiAgQ29sdW1uLFxyXG4gIEltYWdlRGl2LFxyXG4gIE1pbnVzSWNvbixcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUsXHJcbiAgZ2V0X3Bsb3RfZXJyb3IsXHJcbn0gZnJvbSAnLi4vLi4vcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcclxuaW1wb3J0IHsgQ3VzdG9taXphdGlvbiB9IGZyb20gJy4uLy4uLy4uL2N1c3RvbWl6YXRpb24nO1xyXG5pbXBvcnQgeyBab29tZWRQbG90TWVudSB9IGZyb20gJy4uL21lbnUnO1xyXG5pbXBvcnQgeyBQbG90X3BvcnRhbCB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9wb3J0YWwnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcbmltcG9ydCB7IFBsb3RJbWFnZSB9IGZyb20gJy4uLy4uL3Bsb3QvcGxvdEltYWdlJztcclxuLy8gaW1wb3J0IHsgT3ZlcmxheVdpdGhBbm90aGVyUGxvdCB9IGZyb20gJy4uLy4uLy4uL292ZXJsYXlXaXRoQW5vdGhlclBsb3QnO1xyXG5cclxuaW50ZXJmYWNlIFpvb21lZFBsb3RzUHJvcHMge1xyXG4gIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHM7XHJcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgWm9vbWVkUGxvdCA9ICh7XHJcbiAgc2VsZWN0ZWRfcGxvdCxcclxuICBwYXJhbXNfZm9yX2FwaSxcclxufTogWm9vbWVkUGxvdHNQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtjdXN0b21pemF0aW9uUGFyYW1zLCBzZXRDdXN0b21pemF0aW9uUGFyYW1zXSA9IHVzZVN0YXRlPFxyXG4gICAgUGFydGlhbDxTdG9yZT4gJiBDdXN0b21pemVQcm9wc1xyXG4gID4oKTtcclxuICBjb25zdCBbb3BlbkN1c3RvbWl6YXRpb24sIHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBbaXNQb3J0YWxXaW5kb3dPcGVuLCBzZXRJc1BvcnRhbFdpbmRvd09wZW5dID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IFtvcGVuT3ZlcmxheVBsb3RNZW51LCBzZXRPcGVuT3ZlcmxheVBsb3RNZW51XSA9IHVzZVN0YXRlKGZhbHNlKVxyXG5cclxuICBwYXJhbXNfZm9yX2FwaS5jdXN0b21pemVQcm9wcyA9IGN1c3RvbWl6YXRpb25QYXJhbXM7XHJcbiAgY29uc3QgcGxvdF91cmwgPSBnZXRfcGxvdF91cmwocGFyYW1zX2Zvcl9hcGkpO1xyXG5cclxuICBjb25zdCBjb3B5X29mX3BhcmFtcyA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XHJcbiAgY29weV9vZl9wYXJhbXMud2lkdGggPSBNYXRoLnJvdW5kKHdpbmRvdy5pbm5lckhlaWdodCAqIDEuMzMpO1xyXG5cclxuICBjb25zdCB6b29tZWRfcGxvdF91cmwgPSBnZXRfcGxvdF91cmwoY29weV9vZl9wYXJhbXMpO1xyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuXHJcbiAgY29uc3Qgem9vbWVkUGxvdE1lbnVPcHRpb25zID0gW1xyXG4gICAge1xyXG4gICAgICBsYWJlbDogJ09wZW4gaW4gYSBuZXcgdGFiJyxcclxuICAgICAgdmFsdWU6ICdvcGVuX2luX2FfbmV3X3RhYicsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0SXNQb3J0YWxXaW5kb3dPcGVuKHRydWUpLFxyXG4gICAgICBpY29uOiA8RnVsbHNjcmVlbk91dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdDdXN0b21pemUnLFxyXG4gICAgICB2YWx1ZTogJ0N1c3RvbWl6ZScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUodHJ1ZSksXHJcbiAgICAgIGljb246IDxTZXR0aW5nT3V0bGluZWQgLz4sXHJcbiAgICB9LFxyXG4gICAge1xyXG4gICAgICBsYWJlbDogJ092ZXJsYXkgd2l0aCBhbm90aGVyIHBsb3QnLFxyXG4gICAgICB2YWx1ZTogJ0N1c3RvbWl6ZScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0T3Blbk92ZXJsYXlQbG90TWVudSh0cnVlKSxcclxuICAgICAgaWNvbjogPEJsb2NrT3V0bGluZWQgIC8+LFxyXG4gICAgfSxcclxuICBdO1xyXG5cclxuICBjb25zdCB7IGJsaW5rLCB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8U3R5bGVkQ29sIHNwYWNlPXsyfT5cclxuICAgICAgPE92ZXJsYXlXaXRoQW5vdGhlclBsb3RcclxuICAgICAgICB2aXNpYmxlPXtvcGVuT3ZlcmxheVBsb3RNZW51fVxyXG4gICAgICAgIHNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWw9e3NldE9wZW5PdmVybGF5UGxvdE1lbnV9XHJcbiAgICAgIC8+XHJcbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XHJcbiAgICAgIDxQbG90X3BvcnRhbFxyXG4gICAgICAgIGlzUG9ydGFsV2luZG93T3Blbj17aXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHNldElzUG9ydGFsV2luZG93T3Blbj17c2V0SXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHRpdGxlPXtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XHJcbiAgICAgID5cclxuICAgICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG1pbmhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxyXG4gICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgICAge3NlbGVjdGVkX3Bsb3QuZGlzcGxheWVkTmFtZX1cclxuICAgICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRofVxyXG4gICAgICAgICAgICBoZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAgPFBsb3RJbWFnZVxyXG4gICAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17Y29weV9vZl9wYXJhbXN9XHJcbiAgICAgICAgICAgICAgcGxvdD17c2VsZWN0ZWRfcGxvdH1cclxuICAgICAgICAgICAgICBwbG90VVJMPXt6b29tZWRfcGxvdF91cmx9XHJcbiAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgICAgPC9QbG90X3BvcnRhbD5cclxuICAgICAgey8qIFBsb3Qgb3BlbmVkIGluIGEgbmV3IHRhYiAqL31cclxuICAgICAgPEN1c3RvbWl6YXRpb25cclxuICAgICAgICBwbG90X25hbWU9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICBvcGVuPXtvcGVuQ3VzdG9taXphdGlvbn1cclxuICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUoZmFsc2UpfVxyXG4gICAgICAgIHNldEN1c3RvbWl6YXRpb25QYXJhbXM9e3NldEN1c3RvbWl6YXRpb25QYXJhbXN9XHJcbiAgICAgIC8+XHJcbiAgICAgIDxTdHlsZWRQbG90Um93XHJcbiAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbWluaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgPlxyXG4gICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICB7c2VsZWN0ZWRfcGxvdC5kaXNwbGF5ZWROYW1lfVxyXG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgPENvbHVtbiBkaXNwbGF5PVwiZmxleFwiPlxyXG4gICAgICAgICAgPFpvb21lZFBsb3RNZW51IG9wdGlvbnM9e3pvb21lZFBsb3RNZW51T3B0aW9uc30gLz5cclxuICAgICAgICAgIDxNaW51c0ljb25cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4gcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUocXVlcnksIHNlbGVjdGVkX3Bsb3QpfVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbHVtbj5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIGFsaWduaXRlbXM9XCJjZW50ZXJcIlxyXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aH1cclxuICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgICAgZGlzcGxheT1cImZsZXhcIlxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cclxuICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XHJcbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgIHBsb3RVUkw9e3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvSW1hZ2VEaXY+XHJcbiAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgIDwvU3R5bGVkQ29sPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
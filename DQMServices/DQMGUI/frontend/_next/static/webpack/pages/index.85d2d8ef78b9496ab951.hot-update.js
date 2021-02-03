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
  var url = Object(_utils__WEBPACK_IMPORTED_MODULE_12__["getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames"])(query, selected_plot);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJjdXN0b21pemVQcm9wcyIsInBsb3RfdXJsIiwiZ2V0X3Bsb3RfdXJsIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1cmwiLCJnZXRab29tZWRQbG90c1VybEZvck92ZXJsYXlpbmdQbG90c1dpdGhEaWZmZXJlbnROYW1lcyIsInpvb21lZFBsb3RNZW51T3B0aW9ucyIsImxhYmVsIiwidmFsdWUiLCJhY3Rpb24iLCJpY29uIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm5ld19iYWNrX2VuZCIsInVzZUJsaW5rT25VcGRhdGUiLCJibGluayIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJuYW1lIiwidG9TdHJpbmciLCJtb2RlIiwiZ2V0X3Bsb3RfZXJyb3IiLCJyZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7QUFVQTtBQVFBO0FBSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBT08sSUFBTUEsVUFBVSxHQUFHLFNBQWJBLFVBQWEsT0FHRjtBQUFBOztBQUFBOztBQUFBLE1BRnRCQyxhQUVzQixRQUZ0QkEsYUFFc0I7QUFBQSxNQUR0QkMsY0FDc0IsUUFEdEJBLGNBQ3NCOztBQUFBLGtCQUNnQ0Msc0RBQVEsRUFEeEM7QUFBQSxNQUNmQyxtQkFEZTtBQUFBLE1BQ01DLHNCQUROOztBQUFBLG1CQUkrQkYsc0RBQVEsQ0FBQyxLQUFELENBSnZDO0FBQUEsTUFJZkcsaUJBSmU7QUFBQSxNQUlJQyx1QkFKSjs7QUFBQSxtQkFLOEJKLHNEQUFRLENBQUMsS0FBRCxDQUx0QztBQUFBLE1BS2ZLLGtCQUxlO0FBQUEsTUFLS0MscUJBTEw7O0FBT3RCUCxnQkFBYyxDQUFDUSxjQUFmLEdBQWdDTixtQkFBaEM7QUFDQSxNQUFNTyxRQUFRLEdBQUdDLG1FQUFZLENBQUNWLGNBQUQsQ0FBN0I7O0FBQ0EsTUFBTVcsY0FBYyxxQkFBUVgsY0FBUixDQUFwQjs7QUFDQVcsZ0JBQWMsQ0FBQ0MsTUFBZixHQUF3QkMsTUFBTSxDQUFDQyxXQUEvQjtBQUNBSCxnQkFBYyxDQUFDSSxLQUFmLEdBQXVCQyxJQUFJLENBQUNDLEtBQUwsQ0FBV0osTUFBTSxDQUFDQyxXQUFQLEdBQXFCLElBQWhDLENBQXZCO0FBRUEsTUFBTUksZUFBZSxHQUFHUixtRUFBWSxDQUFDQyxjQUFELENBQXBDO0FBRUEsTUFBTVEsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7QUFFRixNQUFNQyxHQUFHLEdBQUdDLHFHQUFxRCxDQUFFRixLQUFGLEVBQVN0QixhQUFULENBQWpFO0FBQ0UsTUFBTXlCLHFCQUFxQixHQUFHLENBQzVCO0FBQ0VDLFNBQUssRUFBRSxtQkFEVDtBQUVFQyxTQUFLLEVBQUUsbUJBRlQ7QUFHRUMsVUFBTSxFQUFFO0FBQUEsYUFBTXBCLHFCQUFxQixDQUFDLElBQUQsQ0FBM0I7QUFBQSxLQUhWO0FBSUVxQixRQUFJLEVBQUUsTUFBQyxvRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBSlIsR0FENEIsRUFPNUI7QUFDRUgsU0FBSyxFQUFFLFdBRFQ7QUFFRUMsU0FBSyxFQUFFLFdBRlQ7QUFHRUMsVUFBTSxFQUFFO0FBQUEsYUFBTXRCLHVCQUF1QixDQUFDLElBQUQsQ0FBN0I7QUFBQSxLQUhWO0FBSUV1QixRQUFJLEVBQUUsTUFBQyxpRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBSlIsR0FQNEIsRUFhNUJDLCtEQUFnQixDQUFDQyxZQUFqQixDQUE4QkEsWUFBOUIsSUFBOEM7QUFDNUNMLFNBQUssRUFBRSwyQkFEcUM7QUFFNUNDLFNBQUssRUFBRSxTQUZxQztBQUc1Q0osT0FBRyxFQUFFQSxHQUh1QztBQUk1Q00sUUFBSSxFQUFFLE1BQUMsK0RBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUpzQyxHQWJsQixDQUE5Qjs7QUFuQnNCLDBCQXVDdUJHLGlGQUFnQixFQXZDdkM7QUFBQSxNQXVDZEMsS0F2Q2MscUJBdUNkQSxLQXZDYztBQUFBLE1BdUNQQyx5QkF2Q08scUJBdUNQQSx5QkF2Q087O0FBeUN0QixTQUNFLE1BQUMsOEVBQUQ7QUFBVyxTQUFLLEVBQUUsQ0FBbEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUVFLE1BQUMsc0VBQUQ7QUFDRSxzQkFBa0IsRUFBRTNCLGtCQUR0QjtBQUVFLHlCQUFxQixFQUFFQyxxQkFGekI7QUFHRSxTQUFLLEVBQUVSLGFBQWEsQ0FBQ21DLElBSHZCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FLRSxNQUFDLGtGQUFEO0FBQ0UsYUFBUyxFQUFFRixLQUFLLENBQUNHLFFBQU4sRUFEYjtBQUVFLGFBQVMsRUFBRSxDQUFDTiwrREFBZ0IsQ0FBQ08sSUFBakIsS0FBMEIsUUFBM0IsRUFBcUNELFFBQXJDLEVBRmI7QUFHRSxhQUFTLEVBQUV4QixjQUFjLENBQUNDLE1BSDVCO0FBSUUsU0FBSywyQkFBRUQsY0FBYyxDQUFDSSxLQUFqQiwwREFBRSxzQkFBc0JvQixRQUF0QixFQUpUO0FBS0Usb0JBQWdCLEVBQUUsS0FBS0EsUUFBTCxFQUxwQjtBQU1FLGFBQVMsRUFBRSxLQUFLQSxRQUFMLEVBTmI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQVFFLE1BQUMsZ0ZBQUQ7QUFBYSxTQUFLLEVBQUVFLDZFQUFjLENBQUN0QyxhQUFELENBQWQsQ0FBOEJvQyxRQUE5QixFQUFwQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dwQyxhQUFhLENBQUNtQyxJQURqQixDQVJGLEVBV0UsTUFBQyw2RUFBRDtBQUNFLE1BQUUsRUFBRW5DLGFBQWEsQ0FBQ21DLElBRHBCO0FBRUUsU0FBSyxFQUFFdkIsY0FBYyxDQUFDSSxLQUZ4QjtBQUdFLFVBQU0sRUFBRUosY0FBYyxDQUFDQyxNQUh6QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0UsTUFBQywwREFBRDtBQUNFLFNBQUssRUFBRW9CLEtBRFQ7QUFFRSxrQkFBYyxFQUFFckIsY0FGbEI7QUFHRSxRQUFJLEVBQUVaLGFBSFI7QUFJRSxXQUFPLEVBQUVtQixlQUpYO0FBS0UsU0FBSyxFQUFFRyxLQUxUO0FBTUUsNkJBQXlCLEVBQUVZLHlCQU43QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBTEYsQ0FYRixDQUxGLENBRkYsRUFtQ0UsTUFBQyw0REFBRDtBQUNFLGFBQVMsRUFBRWxDLGFBQWEsQ0FBQ21DLElBRDNCO0FBRUUsUUFBSSxFQUFFOUIsaUJBRlI7QUFHRSxZQUFRLEVBQUU7QUFBQSxhQUFNQyx1QkFBdUIsQ0FBQyxLQUFELENBQTdCO0FBQUEsS0FIWjtBQUlFLDBCQUFzQixFQUFFRixzQkFKMUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQW5DRixFQXlDRSxNQUFDLGtGQUFEO0FBQ0UsYUFBUyxFQUFFNkIsS0FBSyxDQUFDRyxRQUFOLEVBRGI7QUFFRSxhQUFTLEVBQUUsQ0FBQ04sK0RBQWdCLENBQUNPLElBQWpCLEtBQTBCLFFBQTNCLEVBQXFDRCxRQUFyQyxFQUZiO0FBR0UsYUFBUyxFQUFFbkMsY0FBYyxDQUFDWSxNQUg1QjtBQUlFLFNBQUssMkJBQUVaLGNBQWMsQ0FBQ2UsS0FBakIsMERBQUUsc0JBQXNCb0IsUUFBdEIsRUFKVDtBQUtFLG9CQUFnQixFQUFFLEtBQUtBLFFBQUwsRUFMcEI7QUFNRSxhQUFTLEVBQUUsS0FBS0EsUUFBTCxFQU5iO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FRRSxNQUFDLGdGQUFEO0FBQWEsU0FBSyxFQUFFRSw2RUFBYyxDQUFDdEMsYUFBRCxDQUFkLENBQThCb0MsUUFBOUIsRUFBcEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHcEMsYUFBYSxDQUFDbUMsSUFEakIsQ0FSRixFQVdFLE1BQUMsMkVBQUQ7QUFBUSxXQUFPLEVBQUMsTUFBaEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsb0RBQUQ7QUFBZ0IsV0FBTyxFQUFFVixxQkFBekI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBRUUsTUFBQyw4RUFBRDtBQUNFLFdBQU8sRUFBRTtBQUFBLGFBQU1jLHNGQUF1QixDQUFDakIsS0FBRCxFQUFRdEIsYUFBUixDQUE3QjtBQUFBLEtBRFg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUZGLENBWEYsRUFpQkUsTUFBQyw2RUFBRDtBQUNFLGNBQVUsRUFBQyxRQURiO0FBRUUsTUFBRSxFQUFFQSxhQUFhLENBQUNtQyxJQUZwQjtBQUdFLFNBQUssRUFBRWxDLGNBQWMsQ0FBQ2UsS0FIeEI7QUFJRSxVQUFNLEVBQUVmLGNBQWMsQ0FBQ1ksTUFKekI7QUFLRSxXQUFPLEVBQUMsTUFMVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBT0UsTUFBQywwREFBRDtBQUNFLDZCQUF5QixFQUFFcUIseUJBRDdCO0FBRUUsU0FBSyxFQUFFRCxLQUZUO0FBR0Usa0JBQWMsRUFBRWhDLGNBSGxCO0FBSUUsUUFBSSxFQUFFRCxhQUpSO0FBS0UsV0FBTyxFQUFFVSxRQUxYO0FBTUUsU0FBSyxFQUFFWSxLQU5UO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFQRixDQWpCRixDQXpDRixDQURGO0FBOEVELENBMUhNOztHQUFNdkIsVTtVQWtCSXNCLHFELEVBd0I4QlcseUU7OztLQTFDbENqQyxVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4Ljg1ZDJkOGVmNzhiOTQ5NmFiOTUxLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgRnVsbHNjcmVlbk91dGxpbmVkLCBTZXR0aW5nT3V0bGluZWQsIEJsb2NrT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcbmltcG9ydCB7IFN0b3JlIH0gZnJvbSAnYW50ZC9saWIvZm9ybS9pbnRlcmZhY2UnO1xyXG5cclxuaW1wb3J0IHtcclxuICBnZXRfcGxvdF91cmwsXHJcbiAgZnVuY3Rpb25zX2NvbmZpZyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBQbG90RGF0YVByb3BzLFxyXG4gIFF1ZXJ5UHJvcHMsXHJcbiAgQ3VzdG9taXplUHJvcHMsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZENvbCxcclxuICBQbG90TmFtZUNvbCxcclxuICBTdHlsZWRQbG90Um93LFxyXG4gIENvbHVtbixcclxuICBJbWFnZURpdixcclxuICBNaW51c0ljb24sXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlLFxyXG4gIGdldF9wbG90X2Vycm9yLFxyXG59IGZyb20gJy4uLy4uL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IEN1c3RvbWl6YXRpb24gfSBmcm9tICcuLi8uLi8uLi9jdXN0b21pemF0aW9uJztcclxuaW1wb3J0IHsgWm9vbWVkUGxvdE1lbnUgfSBmcm9tICcuLi9tZW51JztcclxuaW1wb3J0IHsgUGxvdF9wb3J0YWwgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvcG9ydGFsJztcclxuaW1wb3J0IHsgdXNlQmxpbmtPblVwZGF0ZSB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZUJsaW5rT25VcGRhdGUnO1xyXG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi8uLi9wbG90L3Bsb3RJbWFnZSc7XHJcbmltcG9ydCB7IGdldFpvb21lZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzIH0gZnJvbSAnLi4vLi4vLi4vdXRpbHMnO1xyXG5cclxuaW50ZXJmYWNlIFpvb21lZFBsb3RzUHJvcHMge1xyXG4gIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHM7XHJcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgWm9vbWVkUGxvdCA9ICh7XHJcbiAgc2VsZWN0ZWRfcGxvdCxcclxuICBwYXJhbXNfZm9yX2FwaSxcclxufTogWm9vbWVkUGxvdHNQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtjdXN0b21pemF0aW9uUGFyYW1zLCBzZXRDdXN0b21pemF0aW9uUGFyYW1zXSA9IHVzZVN0YXRlPFxyXG4gICAgUGFydGlhbDxTdG9yZT4gJiBDdXN0b21pemVQcm9wc1xyXG4gID4oKTtcclxuICBjb25zdCBbb3BlbkN1c3RvbWl6YXRpb24sIHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBbaXNQb3J0YWxXaW5kb3dPcGVuLCBzZXRJc1BvcnRhbFdpbmRvd09wZW5dID0gdXNlU3RhdGUoZmFsc2UpO1xyXG5cclxuICBwYXJhbXNfZm9yX2FwaS5jdXN0b21pemVQcm9wcyA9IGN1c3RvbWl6YXRpb25QYXJhbXM7XHJcbiAgY29uc3QgcGxvdF91cmwgPSBnZXRfcGxvdF91cmwocGFyYW1zX2Zvcl9hcGkpO1xyXG4gIGNvbnN0IGNvcHlfb2ZfcGFyYW1zID0geyAuLi5wYXJhbXNfZm9yX2FwaSB9O1xyXG4gIGNvcHlfb2ZfcGFyYW1zLmhlaWdodCA9IHdpbmRvdy5pbm5lckhlaWdodDtcclxuICBjb3B5X29mX3BhcmFtcy53aWR0aCA9IE1hdGgucm91bmQod2luZG93LmlubmVySGVpZ2h0ICogMS4zMyk7XHJcblxyXG4gIGNvbnN0IHpvb21lZF9wbG90X3VybCA9IGdldF9wbG90X3VybChjb3B5X29mX3BhcmFtcyk7XHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG5cclxuY29uc3QgdXJsID0gZ2V0Wm9vbWVkUGxvdHNVcmxGb3JPdmVybGF5aW5nUGxvdHNXaXRoRGlmZmVyZW50TmFtZXMoIHF1ZXJ5LCBzZWxlY3RlZF9wbG90KVxyXG4gIGNvbnN0IHpvb21lZFBsb3RNZW51T3B0aW9ucyA9IFtcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdPcGVuIGluIGEgbmV3IHRhYicsXHJcbiAgICAgIHZhbHVlOiAnb3Blbl9pbl9hX25ld190YWInLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHNldElzUG9ydGFsV2luZG93T3Blbih0cnVlKSxcclxuICAgICAgaWNvbjogPEZ1bGxzY3JlZW5PdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgICB7XHJcbiAgICAgIGxhYmVsOiAnQ3VzdG9taXplJyxcclxuICAgICAgdmFsdWU6ICdjdXN0b21pemUnLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KHRydWUpLFxyXG4gICAgICBpY29uOiA8U2V0dGluZ091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCAmJiB7XHJcbiAgICAgIGxhYmVsOiAnT3ZlcmxheSB3aXRoIGFub3RoZXIgcGxvdCcsXHJcbiAgICAgIHZhbHVlOiAnb3ZlcmxheScsXHJcbiAgICAgIHVybDogdXJsLFxyXG4gICAgICBpY29uOiA8QmxvY2tPdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgXTtcclxuICBjb25zdCB7IGJsaW5rLCB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8U3R5bGVkQ29sIHNwYWNlPXsyfT5cclxuICAgICAgey8qIFBsb3Qgb3BlbmVkIGluIGEgbmV3IHRhYiAqL31cclxuICAgICAgPFBsb3RfcG9ydGFsXHJcbiAgICAgICAgaXNQb3J0YWxXaW5kb3dPcGVuPXtpc1BvcnRhbFdpbmRvd09wZW59XHJcbiAgICAgICAgc2V0SXNQb3J0YWxXaW5kb3dPcGVuPXtzZXRJc1BvcnRhbFdpbmRvd09wZW59XHJcbiAgICAgICAgdGl0bGU9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgPlxyXG4gICAgICAgIDxTdHlsZWRQbG90Um93XHJcbiAgICAgICAgICBpc0xvYWRpbmc9e2JsaW5rLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgICAgbWluaGVpZ2h0PXtjb3B5X29mX3BhcmFtcy5oZWlnaHR9XHJcbiAgICAgICAgICB3aWR0aD17Y29weV9vZl9wYXJhbXMud2lkdGg/LnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICA+XHJcbiAgICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxyXG4gICAgICAgICAgICB7c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICAgIDxJbWFnZURpdlxyXG4gICAgICAgICAgICBpZD17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgICAgICB3aWR0aD17Y29weV9vZl9wYXJhbXMud2lkdGh9XHJcbiAgICAgICAgICAgIGhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxyXG4gICAgICAgICAgPlxyXG4gICAgICAgICAgICA8UGxvdEltYWdlXHJcbiAgICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICAgIHBhcmFtc19mb3JfYXBpPXtjb3B5X29mX3BhcmFtc31cclxuICAgICAgICAgICAgICBwbG90PXtzZWxlY3RlZF9wbG90fVxyXG4gICAgICAgICAgICAgIHBsb3RVUkw9e3pvb21lZF9wbG90X3VybH1cclxuICAgICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cclxuICAgICAgICAgICAgLz5cclxuICAgICAgICAgIDwvSW1hZ2VEaXY+XHJcbiAgICAgICAgPC9TdHlsZWRQbG90Um93PlxyXG4gICAgICA8L1Bsb3RfcG9ydGFsPlxyXG4gICAgICB7LyogUGxvdCBvcGVuZWQgaW4gYSBuZXcgdGFiICovfVxyXG4gICAgICA8Q3VzdG9taXphdGlvblxyXG4gICAgICAgIHBsb3RfbmFtZT17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgIG9wZW49e29wZW5DdXN0b21pemF0aW9ufVxyXG4gICAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVDdXN0b21pemF0aW9uTWVudShmYWxzZSl9XHJcbiAgICAgICAgc2V0Q3VzdG9taXphdGlvblBhcmFtcz17c2V0Q3VzdG9taXphdGlvblBhcmFtc31cclxuICAgICAgLz5cclxuICAgICAgPFN0eWxlZFBsb3RSb3dcclxuICAgICAgICBpc0xvYWRpbmc9e2JsaW5rLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgICBtaW5oZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGg/LnRvU3RyaW5nKCl9XHJcbiAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICA8Q29sdW1uIGRpc3BsYXk9XCJmbGV4XCI+XHJcbiAgICAgICAgICA8Wm9vbWVkUGxvdE1lbnUgb3B0aW9ucz17em9vbWVkUGxvdE1lbnVPcHRpb25zfSAvPlxyXG4gICAgICAgICAgPE1pbnVzSWNvblxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZShxdWVyeSwgc2VsZWN0ZWRfcGxvdCl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sdW1uPlxyXG4gICAgICAgIDxJbWFnZURpdlxyXG4gICAgICAgICAgYWxpZ25pdGVtcz1cImNlbnRlclwiXHJcbiAgICAgICAgICBpZD17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRofVxyXG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgICBkaXNwbGF5PVwiZmxleFwiXHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPFBsb3RJbWFnZVxyXG4gICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgIHBhcmFtc19mb3JfYXBpPXtwYXJhbXNfZm9yX2FwaX1cclxuICAgICAgICAgICAgcGxvdD17c2VsZWN0ZWRfcGxvdH1cclxuICAgICAgICAgICAgcGxvdFVSTD17cGxvdF91cmx9XHJcbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9JbWFnZURpdj5cclxuICAgICAgPC9TdHlsZWRQbG90Um93PlxyXG4gICAgPC9TdHlsZWRDb2w+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
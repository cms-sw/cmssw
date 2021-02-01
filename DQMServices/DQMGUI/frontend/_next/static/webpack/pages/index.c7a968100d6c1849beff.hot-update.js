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
        lineNumber: 65,
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
        lineNumber: 71,
        columnNumber: 13
      }
    })
  }, _config_config__WEBPACK_IMPORTED_MODULE_4__["functions_config"].new_back_end.new_back_end && {
    label: 'Overlay with another plot',
    value: 'overlay',
    action: function action() {
      var basePath = router.basePath;
      var page = 'plotsLocalOverlay';
      var run = 'run_number=' + query.run_number;
      var dataset = 'dataset_name=' + query.dataset_name;
      var path = 'folders_path=' + selected_plot.path;
      var plot_name = 'plot_name=' + selected_plot.name;
      var baseURL = [basePath, page].join('/');
      var queryURL = [run, dataset, path, plot_name].join('&');
      var plotsLocalOverlayURL = [baseURL, queryURL].join('?');
      return plotsLocalOverlayURL;
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 88,
        columnNumber: 14
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
      lineNumber: 94,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_9__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
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
      lineNumber: 101,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 109,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 112,
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
      lineNumber: 117,
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
      lineNumber: 129,
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
      lineNumber: 135,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 143,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 146,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 147,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 148,
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
      lineNumber: 152,
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
      lineNumber: 159,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJjdXN0b21pemVQcm9wcyIsInBsb3RfdXJsIiwiZ2V0X3Bsb3RfdXJsIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ6b29tZWRQbG90TWVudU9wdGlvbnMiLCJsYWJlbCIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJiYXNlUGF0aCIsInBhZ2UiLCJydW4iLCJydW5fbnVtYmVyIiwiZGF0YXNldCIsImRhdGFzZXRfbmFtZSIsInBhdGgiLCJwbG90X25hbWUiLCJuYW1lIiwiYmFzZVVSTCIsImpvaW4iLCJxdWVyeVVSTCIsInBsb3RzTG9jYWxPdmVybGF5VVJMIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInRvU3RyaW5nIiwibW9kZSIsImdldF9wbG90X2Vycm9yIiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7QUFVQTtBQVFBO0FBSUE7QUFD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SUFEakIsQ0FSRixFQVdFLE1BQUMsMkVBQUQ7QUFBUSxXQUFPLEVBQUMsTUFBaEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsb0RBQUQ7QUFBZ0IsV0FBTyxFQUFFZixxQkFBekI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBRUUsTUFBQyw4RUFBRDtBQUNFLFdBQU8sRUFBRTtBQUFBLGFBQU0wQixzRkFBdUIsQ0FBQzNCLEtBQUQsRUFBUXRCLGFBQVIsQ0FBN0I7QUFBQSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFGRixDQVhGLEVBaUJFLE1BQUMsNkVBQUQ7QUFDRSxjQUFVLEVBQUMsUUFEYjtBQUVFLE1BQUUsRUFBRUEsYUFBYSxDQUFDc0MsSUFGcEI7QUFHRSxTQUFLLEVBQUVyQyxjQUFjLENBQUNlLEtBSHhCO0FBSUUsVUFBTSxFQUFFZixjQUFjLENBQUNZLE1BSnpCO0FBS0UsV0FBTyxFQUFDLE1BTFY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU9FLE1BQUMsMERBQUQ7QUFDRSw2QkFBeUIsRUFBRWdDLHlCQUQ3QjtBQUVFLFNBQUssRUFBRUQsS0FGVDtBQUdFLGtCQUFjLEVBQUUzQyxjQUhsQjtBQUlFLFFBQUksRUFBRUQsYUFKUjtBQUtFLFdBQU8sRUFBRVUsUUFMWDtBQU1FLFNBQUssRUFBRVksS0FOVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBUEYsQ0FqQkYsQ0F6Q0YsQ0FERjtBQThFRCxDQXBJTTs7R0FBTXZCLFU7VUFrQklzQixxRCxFQWtDOEJzQix5RTs7O0tBcERsQzVDLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguYzdhOTY4MTAwZDZjMTg0OWJlZmYuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGdWxsc2NyZWVuT3V0bGluZWQsIFNldHRpbmdPdXRsaW5lZCwgQmxvY2tPdXRsaW5lZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuaW1wb3J0IHsgU3RvcmUgfSBmcm9tICdhbnRkL2xpYi9mb3JtL2ludGVyZmFjZSc7XHJcblxyXG5pbXBvcnQge1xyXG4gIGdldF9wbG90X3VybCxcclxuICBmdW5jdGlvbnNfY29uZmlnLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFBsb3REYXRhUHJvcHMsXHJcbiAgUXVlcnlQcm9wcyxcclxuICBDdXN0b21pemVQcm9wcyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkQ29sLFxyXG4gIFBsb3ROYW1lQ29sLFxyXG4gIFN0eWxlZFBsb3RSb3csXHJcbiAgQ29sdW1uLFxyXG4gIEltYWdlRGl2LFxyXG4gIE1pbnVzSWNvbixcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUsXHJcbiAgZ2V0X3Bsb3RfZXJyb3IsXHJcbn0gZnJvbSAnLi4vLi4vcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcclxuaW1wb3J0IHsgQ3VzdG9taXphdGlvbiB9IGZyb20gJy4uLy4uLy4uL2N1c3RvbWl6YXRpb24nO1xyXG5pbXBvcnQgeyBab29tZWRQbG90TWVudSB9IGZyb20gJy4uL21lbnUnO1xyXG5pbXBvcnQgeyBQbG90X3BvcnRhbCB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9wb3J0YWwnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcbmltcG9ydCB7IFBsb3RJbWFnZSB9IGZyb20gJy4uLy4uL3Bsb3QvcGxvdEltYWdlJztcclxuXHJcbmludGVyZmFjZSBab29tZWRQbG90c1Byb3BzIHtcclxuICBzZWxlY3RlZF9wbG90OiBQbG90RGF0YVByb3BzO1xyXG4gIHBhcmFtc19mb3JfYXBpOiBQYXJhbXNGb3JBcGlQcm9wcztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3QgPSAoe1xyXG4gIHNlbGVjdGVkX3Bsb3QsXHJcbiAgcGFyYW1zX2Zvcl9hcGksXHJcbn06IFpvb21lZFBsb3RzUHJvcHMpID0+IHtcclxuICBjb25zdCBbY3VzdG9taXphdGlvblBhcmFtcywgc2V0Q3VzdG9taXphdGlvblBhcmFtc10gPSB1c2VTdGF0ZTxcclxuICAgIFBhcnRpYWw8U3RvcmU+ICYgQ3VzdG9taXplUHJvcHNcclxuICA+KCk7XHJcbiAgY29uc3QgW29wZW5DdXN0b21pemF0aW9uLCB0b2dnbGVDdXN0b21pemF0aW9uTWVudV0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3QgW2lzUG9ydGFsV2luZG93T3Blbiwgc2V0SXNQb3J0YWxXaW5kb3dPcGVuXSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuXHJcbiAgcGFyYW1zX2Zvcl9hcGkuY3VzdG9taXplUHJvcHMgPSBjdXN0b21pemF0aW9uUGFyYW1zO1xyXG4gIGNvbnN0IHBsb3RfdXJsID0gZ2V0X3Bsb3RfdXJsKHBhcmFtc19mb3JfYXBpKTtcclxuICBjb25zdCBjb3B5X29mX3BhcmFtcyA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuICBjb3B5X29mX3BhcmFtcy5oZWlnaHQgPSB3aW5kb3cuaW5uZXJIZWlnaHQ7XHJcbiAgY29weV9vZl9wYXJhbXMud2lkdGggPSBNYXRoLnJvdW5kKHdpbmRvdy5pbm5lckhlaWdodCAqIDEuMzMpO1xyXG5cclxuICBjb25zdCB6b29tZWRfcGxvdF91cmwgPSBnZXRfcGxvdF91cmwoY29weV9vZl9wYXJhbXMpO1xyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuXHJcbiAgY29uc3Qgem9vbWVkUGxvdE1lbnVPcHRpb25zID0gW1xyXG4gICAge1xyXG4gICAgICBsYWJlbDogJ09wZW4gaW4gYSBuZXcgdGFiJyxcclxuICAgICAgdmFsdWU6ICdvcGVuX2luX2FfbmV3X3RhYicsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gc2V0SXNQb3J0YWxXaW5kb3dPcGVuKHRydWUpLFxyXG4gICAgICBpY29uOiA8RnVsbHNjcmVlbk91dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdDdXN0b21pemUnLFxyXG4gICAgICB2YWx1ZTogJ2N1c3RvbWl6ZScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUodHJ1ZSksXHJcbiAgICAgIGljb246IDxTZXR0aW5nT3V0bGluZWQgLz4sXHJcbiAgICB9LFxyXG4gICAgZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kICYmIHtcclxuICAgICAgbGFiZWw6ICdPdmVybGF5IHdpdGggYW5vdGhlciBwbG90JyxcclxuICAgICAgdmFsdWU6ICdvdmVybGF5JyxcclxuICAgICAgYWN0aW9uOiAoKSA9PiB7XHJcbiAgICAgICAgY29uc3QgYmFzZVBhdGggPSByb3V0ZXIuYmFzZVBhdGhcclxuICAgICAgICBjb25zdCBwYWdlID0gJ3Bsb3RzTG9jYWxPdmVybGF5J1xyXG4gICAgICAgIGNvbnN0IHJ1biA9ICdydW5fbnVtYmVyPScgKyBxdWVyeS5ydW5fbnVtYmVyIGFzIHN0cmluZ1xyXG4gICAgICAgIGNvbnN0IGRhdGFzZXQgPSdkYXRhc2V0X25hbWU9JyArIHF1ZXJ5LmRhdGFzZXRfbmFtZSAgYXMgc3RyaW5nXHJcbiAgICAgICAgY29uc3QgcGF0aCA9ICdmb2xkZXJzX3BhdGg9JyArIHNlbGVjdGVkX3Bsb3QucGF0aCBcclxuICAgICAgICBjb25zdCBwbG90X25hbWUgPSAncGxvdF9uYW1lPScgKyBzZWxlY3RlZF9wbG90Lm5hbWVcclxuICAgICAgICBjb25zdCBiYXNlVVJMID0gW2Jhc2VQYXRoLCBwYWdlXS5qb2luKCcvJylcclxuICAgICAgICBjb25zdCBxdWVyeVVSTCA9IFsgcnVuICwgZGF0YXNldCwgcGF0aCwgcGxvdF9uYW1lXS5qb2luKCcmJylcclxuICAgICAgICBjb25zdCBwbG90c0xvY2FsT3ZlcmxheVVSTCA9IFtiYXNlVVJMLCBxdWVyeVVSTF0uam9pbignPycpXHJcbiAgICAgICAgcmV0dXJuIHBsb3RzTG9jYWxPdmVybGF5VVJMXHJcbiAgICAgIH0sXHJcbiAgICAgICBpY29uOiA8QmxvY2tPdXRsaW5lZCAgLz4sXHJcbiAgICB9LFxyXG4gIF07XHJcbiAgY29uc3QgeyBibGluaywgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZENvbCBzcGFjZT17Mn0+XHJcbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XHJcbiAgICAgIDxQbG90X3BvcnRhbFxyXG4gICAgICAgIGlzUG9ydGFsV2luZG93T3Blbj17aXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHNldElzUG9ydGFsV2luZG93T3Blbj17c2V0SXNQb3J0YWxXaW5kb3dPcGVufVxyXG4gICAgICAgIHRpdGxlPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgID5cclxuICAgICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG1pbmhlaWdodD17Y29weV9vZl9wYXJhbXMuaGVpZ2h0fVxyXG4gICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgICAgaXNfcGxvdF9zZWxlY3RlZD17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgICAge3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgICAgd2lkdGg9e2NvcHlfb2ZfcGFyYW1zLndpZHRofVxyXG4gICAgICAgICAgICBoZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAgPFBsb3RJbWFnZVxyXG4gICAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17Y29weV9vZl9wYXJhbXN9XHJcbiAgICAgICAgICAgICAgcGxvdD17c2VsZWN0ZWRfcGxvdH1cclxuICAgICAgICAgICAgICBwbG90VVJMPXt6b29tZWRfcGxvdF91cmx9XHJcbiAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgICAgPC9QbG90X3BvcnRhbD5cclxuICAgICAgey8qIFBsb3Qgb3BlbmVkIGluIGEgbmV3IHRhYiAqL31cclxuICAgICAgPEN1c3RvbWl6YXRpb25cclxuICAgICAgICBwbG90X25hbWU9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICBvcGVuPXtvcGVuQ3VzdG9taXphdGlvbn1cclxuICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUoZmFsc2UpfVxyXG4gICAgICAgIHNldEN1c3RvbWl6YXRpb25QYXJhbXM9e3NldEN1c3RvbWl6YXRpb25QYXJhbXN9XHJcbiAgICAgIC8+XHJcbiAgICAgIDxTdHlsZWRQbG90Um93XHJcbiAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbWluaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgPlxyXG4gICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICB7c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgPENvbHVtbiBkaXNwbGF5PVwiZmxleFwiPlxyXG4gICAgICAgICAgPFpvb21lZFBsb3RNZW51IG9wdGlvbnM9e3pvb21lZFBsb3RNZW51T3B0aW9uc30gLz5cclxuICAgICAgICAgIDxNaW51c0ljb25cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4gcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUocXVlcnksIHNlbGVjdGVkX3Bsb3QpfVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbHVtbj5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIGFsaWduaXRlbXM9XCJjZW50ZXJcIlxyXG4gICAgICAgICAgaWQ9e3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aH1cclxuICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgICAgZGlzcGxheT1cImZsZXhcIlxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1cclxuICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICBwYXJhbXNfZm9yX2FwaT17cGFyYW1zX2Zvcl9hcGl9XHJcbiAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgIHBsb3RVUkw9e3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvSW1hZ2VEaXY+XHJcbiAgICAgIDwvU3R5bGVkUGxvdFJvdz5cclxuICAgIDwvU3R5bGVkQ29sPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
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
      var run_number = query.run_number;
      var dataset_name = query.dataset_name;
      var folders_path = selected_plot.path;
      var plot_name = selected_plot.name;
      var queryProps = {
        path: page,
        query: {
          run_number: run_number,
          dataset_name: dataset_name,
          folders_path: folders_path,
          plot_name: plot_name
        }
      };
    },
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["BlockOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 94,
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
      lineNumber: 100,
      columnNumber: 5
    }
  }, __jsx(_containers_display_portal__WEBPACK_IMPORTED_MODULE_9__["Plot_portal"], {
    isPortalWindowOpen: isPortalWindowOpen,
    setIsPortalWindowOpen: setIsPortalWindowOpen,
    title: selected_plot.name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 102,
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
      lineNumber: 107,
      columnNumber: 9
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 115,
      columnNumber: 11
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ImageDiv"], {
    id: selected_plot.name,
    width: copy_of_params.width,
    height: copy_of_params.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
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
      lineNumber: 123,
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
      lineNumber: 135,
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
      lineNumber: 141,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 149,
      columnNumber: 9
    }
  }, selected_plot.name), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Column"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 152,
      columnNumber: 9
    }
  }, __jsx(_menu__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotMenu"], {
    options: zoomedPlotMenuOptions,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 153,
      columnNumber: 11
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["MinusIcon"], {
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["removePlotFromRightSide"])(query, selected_plot);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 154,
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
      lineNumber: 158,
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
      lineNumber: 165,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRQbG90LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwidXNlU3RhdGUiLCJjdXN0b21pemF0aW9uUGFyYW1zIiwic2V0Q3VzdG9taXphdGlvblBhcmFtcyIsIm9wZW5DdXN0b21pemF0aW9uIiwidG9nZ2xlQ3VzdG9taXphdGlvbk1lbnUiLCJpc1BvcnRhbFdpbmRvd09wZW4iLCJzZXRJc1BvcnRhbFdpbmRvd09wZW4iLCJjdXN0b21pemVQcm9wcyIsInBsb3RfdXJsIiwiZ2V0X3Bsb3RfdXJsIiwiY29weV9vZl9wYXJhbXMiLCJoZWlnaHQiLCJ3aW5kb3ciLCJpbm5lckhlaWdodCIsIndpZHRoIiwiTWF0aCIsInJvdW5kIiwiem9vbWVkX3Bsb3RfdXJsIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ6b29tZWRQbG90TWVudU9wdGlvbnMiLCJsYWJlbCIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJiYXNlUGF0aCIsInBhZ2UiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwiZm9sZGVyc19wYXRoIiwicGF0aCIsInBsb3RfbmFtZSIsIm5hbWUiLCJxdWVyeVByb3BzIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInRvU3RyaW5nIiwibW9kZSIsImdldF9wbG90X2Vycm9yIiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7QUFVQTtBQVFBO0FBSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQU9PLElBQU1BLFVBQVUsR0FBRyxT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QUMsOEVBQUQ7QUFDRSxXQUFPLEVBQUU7QUFBQSxhQUFNc0Isc0ZBQXVCLENBQUN2QixLQUFELEVBQVF0QixhQUFSLENBQTdCO0FBQUEsS0FEWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRkYsQ0FYRixFQWlCRSxNQUFDLDZFQUFEO0FBQ0UsY0FBVSxFQUFDLFFBRGI7QUFFRSxNQUFFLEVBQUVBLGFBQWEsQ0FBQ3FDLElBRnBCO0FBR0UsU0FBSyxFQUFFcEMsY0FBYyxDQUFDZSxLQUh4QjtBQUlFLFVBQU0sRUFBRWYsY0FBYyxDQUFDWSxNQUp6QjtBQUtFLFdBQU8sRUFBQyxNQUxWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FPRSxNQUFDLDBEQUFEO0FBQ0UsNkJBQXlCLEVBQUU0Qix5QkFEN0I7QUFFRSxTQUFLLEVBQUVELEtBRlQ7QUFHRSxrQkFBYyxFQUFFdkMsY0FIbEI7QUFJRSxRQUFJLEVBQUVELGFBSlI7QUFLRSxXQUFPLEVBQUVVLFFBTFg7QUFNRSxTQUFLLEVBQUVZLEtBTlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQVBGLENBakJGLENBekNGLENBREY7QUE4RUQsQ0ExSU07O0dBQU12QixVO1VBa0JJc0IscUQsRUF3QzhCa0IseUU7OztLQTFEbEN4QyxVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmUzODlmYjZkMDIxNWJjMGRkNWNlLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgRnVsbHNjcmVlbk91dGxpbmVkLCBTZXR0aW5nT3V0bGluZWQsIEJsb2NrT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcbmltcG9ydCB7IFN0b3JlIH0gZnJvbSAnYW50ZC9saWIvZm9ybS9pbnRlcmZhY2UnO1xyXG5cclxuaW1wb3J0IHtcclxuICBnZXRfcGxvdF91cmwsXHJcbiAgZnVuY3Rpb25zX2NvbmZpZyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBQbG90RGF0YVByb3BzLFxyXG4gIFF1ZXJ5UHJvcHMsXHJcbiAgQ3VzdG9taXplUHJvcHMsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZENvbCxcclxuICBQbG90TmFtZUNvbCxcclxuICBTdHlsZWRQbG90Um93LFxyXG4gIENvbHVtbixcclxuICBJbWFnZURpdixcclxuICBNaW51c0ljb24sXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlLFxyXG4gIGdldF9wbG90X2Vycm9yLFxyXG59IGZyb20gJy4uLy4uL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IEN1c3RvbWl6YXRpb24gfSBmcm9tICcuLi8uLi8uLi9jdXN0b21pemF0aW9uJztcclxuaW1wb3J0IHsgWm9vbWVkUGxvdE1lbnUgfSBmcm9tICcuLi9tZW51JztcclxuaW1wb3J0IHsgUGxvdF9wb3J0YWwgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvcG9ydGFsJztcclxuaW1wb3J0IHsgdXNlQmxpbmtPblVwZGF0ZSB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZUJsaW5rT25VcGRhdGUnO1xyXG5pbXBvcnQgeyBQbG90SW1hZ2UgfSBmcm9tICcuLi8uLi9wbG90L3Bsb3RJbWFnZSc7XHJcblxyXG5pbnRlcmZhY2UgWm9vbWVkUGxvdHNQcm9wcyB7XHJcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcclxuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRQbG90ID0gKHtcclxuICBzZWxlY3RlZF9wbG90LFxyXG4gIHBhcmFtc19mb3JfYXBpLFxyXG59OiBab29tZWRQbG90c1Byb3BzKSA9PiB7XHJcbiAgY29uc3QgW2N1c3RvbWl6YXRpb25QYXJhbXMsIHNldEN1c3RvbWl6YXRpb25QYXJhbXNdID0gdXNlU3RhdGU8XHJcbiAgICBQYXJ0aWFsPFN0b3JlPiAmIEN1c3RvbWl6ZVByb3BzXHJcbiAgPigpO1xyXG4gIGNvbnN0IFtvcGVuQ3VzdG9taXphdGlvbiwgdG9nZ2xlQ3VzdG9taXphdGlvbk1lbnVdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IFtpc1BvcnRhbFdpbmRvd09wZW4sIHNldElzUG9ydGFsV2luZG93T3Blbl0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIHBhcmFtc19mb3JfYXBpLmN1c3RvbWl6ZVByb3BzID0gY3VzdG9taXphdGlvblBhcmFtcztcclxuICBjb25zdCBwbG90X3VybCA9IGdldF9wbG90X3VybChwYXJhbXNfZm9yX2FwaSk7XHJcbiAgY29uc3QgY29weV9vZl9wYXJhbXMgPSB7IC4uLnBhcmFtc19mb3JfYXBpIH07XHJcbiAgY29weV9vZl9wYXJhbXMuaGVpZ2h0ID0gd2luZG93LmlubmVySGVpZ2h0O1xyXG4gIGNvcHlfb2ZfcGFyYW1zLndpZHRoID0gTWF0aC5yb3VuZCh3aW5kb3cuaW5uZXJIZWlnaHQgKiAxLjMzKTtcclxuXHJcbiAgY29uc3Qgem9vbWVkX3Bsb3RfdXJsID0gZ2V0X3Bsb3RfdXJsKGNvcHlfb2ZfcGFyYW1zKTtcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IHpvb21lZFBsb3RNZW51T3B0aW9ucyA9IFtcclxuICAgIHtcclxuICAgICAgbGFiZWw6ICdPcGVuIGluIGEgbmV3IHRhYicsXHJcbiAgICAgIHZhbHVlOiAnb3Blbl9pbl9hX25ld190YWInLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHNldElzUG9ydGFsV2luZG93T3Blbih0cnVlKSxcclxuICAgICAgaWNvbjogPEZ1bGxzY3JlZW5PdXRsaW5lZCAvPixcclxuICAgIH0sXHJcbiAgICB7XHJcbiAgICAgIGxhYmVsOiAnQ3VzdG9taXplJyxcclxuICAgICAgdmFsdWU6ICdjdXN0b21pemUnLFxyXG4gICAgICBhY3Rpb246ICgpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KHRydWUpLFxyXG4gICAgICBpY29uOiA8U2V0dGluZ091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCAmJiB7XHJcbiAgICAgIGxhYmVsOiAnT3ZlcmxheSB3aXRoIGFub3RoZXIgcGxvdCcsXHJcbiAgICAgIHZhbHVlOiAnb3ZlcmxheScsXHJcbiAgICAgIGFjdGlvbjogKCkgPT4ge1xyXG4gICAgICAgIGNvbnN0IGJhc2VQYXRoID0gcm91dGVyLmJhc2VQYXRoXHJcbiAgICAgICAgY29uc3QgcGFnZSA9ICdwbG90c0xvY2FsT3ZlcmxheSdcclxuICAgICAgICBjb25zdCBydW5fbnVtYmVyID0gcXVlcnkucnVuX251bWJlciBhcyBzdHJpbmdcclxuICAgICAgICBjb25zdCBkYXRhc2V0X25hbWUgPSBxdWVyeS5kYXRhc2V0X25hbWUgYXMgc3RyaW5nXHJcbiAgICAgICAgY29uc3QgZm9sZGVyc19wYXRoID0gc2VsZWN0ZWRfcGxvdC5wYXRoXHJcbiAgICAgICAgY29uc3QgcGxvdF9uYW1lID0gc2VsZWN0ZWRfcGxvdC5uYW1lXHJcblxyXG4gICAgICAgIGNvbnN0IHF1ZXJ5UHJvcHMgPSB7XHJcbiAgICAgICAgICBwYXRoOiBwYWdlLFxyXG4gICAgICAgICAgcXVlcnk6IHtcclxuICAgICAgICAgICAgcnVuX251bWJlcixcclxuICAgICAgICAgICAgZGF0YXNldF9uYW1lLFxyXG4gICAgICAgICAgICBmb2xkZXJzX3BhdGgsXHJcbiAgICAgICAgICAgIHBsb3RfbmFtZSxcclxuICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgIH0sXHJcbiAgICAgIGljb246IDxCbG9ja091dGxpbmVkIC8+LFxyXG4gICAgfSxcclxuICBdO1xyXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICB7LyogUGxvdCBvcGVuZWQgaW4gYSBuZXcgdGFiICovfVxyXG4gICAgICA8UGxvdF9wb3J0YWxcclxuICAgICAgICBpc1BvcnRhbFdpbmRvd09wZW49e2lzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICBzZXRJc1BvcnRhbFdpbmRvd09wZW49e3NldElzUG9ydGFsV2luZG93T3Blbn1cclxuICAgICAgICB0aXRsZT17c2VsZWN0ZWRfcGxvdC5uYW1lfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFN0eWxlZFBsb3RSb3dcclxuICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICBtaW5oZWlnaHQ9e2NvcHlfb2ZfcGFyYW1zLmhlaWdodH1cclxuICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICAgIG5vcG9pbnRlcj17dHJ1ZS50b1N0cmluZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICAgIHtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICAgIHdpZHRoPXtjb3B5X29mX3BhcmFtcy53aWR0aH1cclxuICAgICAgICAgICAgaGVpZ2h0PXtjb3B5X29mX3BhcmFtcy5oZWlnaHR9XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIDxQbG90SW1hZ2VcclxuICAgICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e2NvcHlfb2ZfcGFyYW1zfVxyXG4gICAgICAgICAgICAgIHBsb3Q9e3NlbGVjdGVkX3Bsb3R9XHJcbiAgICAgICAgICAgICAgcGxvdFVSTD17em9vbWVkX3Bsb3RfdXJsfVxyXG4gICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufVxyXG4gICAgICAgICAgICAvPlxyXG4gICAgICAgICAgPC9JbWFnZURpdj5cclxuICAgICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICAgIDwvUGxvdF9wb3J0YWw+XHJcbiAgICAgIHsvKiBQbG90IG9wZW5lZCBpbiBhIG5ldyB0YWIgKi99XHJcbiAgICAgIDxDdXN0b21pemF0aW9uXHJcbiAgICAgICAgcGxvdF9uYW1lPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgb3Blbj17b3BlbkN1c3RvbWl6YXRpb259XHJcbiAgICAgICAgb25DYW5jZWw9eygpID0+IHRvZ2dsZUN1c3RvbWl6YXRpb25NZW51KGZhbHNlKX1cclxuICAgICAgICBzZXRDdXN0b21pemF0aW9uUGFyYW1zPXtzZXRDdXN0b21pemF0aW9uUGFyYW1zfVxyXG4gICAgICAvPlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgID5cclxuICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxyXG4gICAgICAgICAge3NlbGVjdGVkX3Bsb3QubmFtZX1cclxuICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgIDxDb2x1bW4gZGlzcGxheT1cImZsZXhcIj5cclxuICAgICAgICAgIDxab29tZWRQbG90TWVudSBvcHRpb25zPXt6b29tZWRQbG90TWVudU9wdGlvbnN9IC8+XHJcbiAgICAgICAgICA8TWludXNJY29uXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Db2x1bW4+XHJcbiAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcclxuICAgICAgICAgIGlkPXtzZWxlY3RlZF9wbG90Lm5hbWV9XHJcbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XHJcbiAgICAgICAgICBoZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICAgIGRpc3BsYXk9XCJmbGV4XCJcclxuICAgICAgICA+XHJcbiAgICAgICAgICA8UGxvdEltYWdlXHJcbiAgICAgICAgICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW49e3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59XHJcbiAgICAgICAgICAgIGJsaW5rPXtibGlua31cclxuICAgICAgICAgICAgcGFyYW1zX2Zvcl9hcGk9e3BhcmFtc19mb3JfYXBpfVxyXG4gICAgICAgICAgICBwbG90PXtzZWxlY3RlZF9wbG90fVxyXG4gICAgICAgICAgICBwbG90VVJMPXtwbG90X3VybH1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0ltYWdlRGl2PlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
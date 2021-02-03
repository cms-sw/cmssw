webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotsWithLayouts/styledComponents.ts":
/*!********************************************************************!*\
  !*** ./components/plots/plot/plotsWithLayouts/styledComponents.ts ***!
  \********************************************************************/
/*! exports provided: ParentWrapper, LayoutName, LayoutWrapper, PlotWrapper */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ParentWrapper", function() { return ParentWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LayoutName", function() { return LayoutName; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LayoutWrapper", function() { return LayoutWrapper; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotWrapper", function() { return PlotWrapper; });
/* harmony import */ var styled_components__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! styled-components */ "./node_modules/styled-components/dist/styled-components.browser.esm.js");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../../../styles/theme */ "./styles/theme.ts");


var keyframe_for_updates_plots = Object(styled_components__WEBPACK_IMPORTED_MODULE_0__["keyframes"])(["0%{background:", ";color:", ";}100%{background:", ";s}"], _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.main, _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.common.white, _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
var ParentWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__ParentWrapper",
  componentId: "qjilkp-0"
})(["width:", "px;height:", "px;justify-content:center;margin:4px;background:", ";display:grid;align-items:end;padding:8px;animation-iteration-count:1;animation-duration:1s;animation-name:", ";"], function (props) {
  return props.size.w + 30 + (props.plotsAmount ? props.plotsAmount : 4 * 4);
}, function (props) {
  return props.size.h + 40 + (props.plotsAmount ? props.plotsAmount : 4 * 4);
}, function (props) {
  return props.isPlotSelected === 'true' ? _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light : _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light;
}, function (props) {
  return props.isLoading === 'true' && props.animation === 'true' ? keyframe_for_updates_plots : '';
});
var LayoutName = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__LayoutName",
  componentId: "qjilkp-1"
})(["padding-bottom:4;color:", ";font-weight:", ";word-break:break-word;"], function (props) {
  return props.error === 'true' ? _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.notification.error : _styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.common.black;
}, function (props) {
  return props.isPlotSelected === 'true' ? 'bold' : '';
});
var LayoutWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__LayoutWrapper",
  componentId: "qjilkp-2"
})(["display:grid;grid-template-columns:", ";justify-content:center;"], function (props) {
  return props.auto;
});
var PlotWrapper = styled_components__WEBPACK_IMPORTED_MODULE_0__["default"].div.withConfig({
  displayName: "styledComponents__PlotWrapper",
  componentId: "qjilkp-3"
})(["justify-content:center;border:", ";align-items:center;width:", ";height:", ";;cursor:pointer;padding:4px;align-self:center;justify-self:baseline;cursor:", ";"], function (props) {
  return props.plotSelected ? "4px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.secondary.light) : "2px solid ".concat(_styles_theme__WEBPACK_IMPORTED_MODULE_1__["theme"].colors.primary.light);
}, function (props) {
  return props.width ? "calc(".concat(props.width, "+8px)") : 'fit-content';
}, function (props) {
  return props.height ? "calc(".concat(props.height, "+8px)") : 'fit-content';
}, function (props) {
  return props.plotSelected ? 'zoom-out' : 'zoom-in';
});

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

/***/ "./config/config.ts":
/*!**************************!*\
  !*** ./config/config.ts ***!
  \**************************/
/*! exports provided: functions_config, root_url, mode, service_title, get_folders_and_plots_new_api, get_folders_and_plots_new_api_with_live_mode, get_folders_and_plots_old_api, get_run_list_by_search_old_api, get_run_list_by_search_new_api, get_run_list_by_search_new_api_with_no_older_than, get_plot_url, get_plot_with_overlay, get_overlaied_plots_urls, get_plot_with_overlay_new_api, get_jroot_plot, getLumisections, get_the_latest_runs */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(process, module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "functions_config", function() { return functions_config; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "root_url", function() { return root_url; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "mode", function() { return mode; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "service_title", function() { return service_title; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_folders_and_plots_new_api", function() { return get_folders_and_plots_new_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_folders_and_plots_new_api_with_live_mode", function() { return get_folders_and_plots_new_api_with_live_mode; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_folders_and_plots_old_api", function() { return get_folders_and_plots_old_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_run_list_by_search_old_api", function() { return get_run_list_by_search_old_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_run_list_by_search_new_api", function() { return get_run_list_by_search_new_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_run_list_by_search_new_api_with_no_older_than", function() { return get_run_list_by_search_new_api_with_no_older_than; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_plot_url", function() { return get_plot_url; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_plot_with_overlay", function() { return get_plot_with_overlay; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_overlaied_plots_urls", function() { return get_overlaied_plots_urls; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_plot_with_overlay_new_api", function() { return get_plot_with_overlay_new_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_jroot_plot", function() { return get_jroot_plot; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getLumisections", function() { return getLumisections; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_the_latest_runs", function() { return get_the_latest_runs; });
/* harmony import */ var _components_constants__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../components/constants */ "./components/constants.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils */ "./config/utils.ts");


var config = {
  development: {
    root_url: 'http://localhost:8081/',
    title: 'Development'
  },
  production: {
    // root_url: `https://dqm-gui.web.cern.ch/api/dqm/offline/`,
    root_url: 'http://localhost:8081/',
    // root_url: `${getPathName()}`,
    title: 'Online-playback'
  }
};
var new_env_variable = "true" === 'true';
var layout_env_variable = "true" === 'true';
var latest_runs_env_variable = "true" === 'true';
var lumis_env_variable = process.env.LUMIS === 'true';
var functions_config = {
  new_back_end: {
    new_back_end: new_env_variable || false,
    lumisections_on: lumis_env_variable && new_env_variable || false,
    layouts: layout_env_variable && new_env_variable || false,
    latest_runs: latest_runs_env_variable && new_env_variable || false
  },
  mode: process.env.MODE || 'OFFLINE'
};
var root_url = config["development" || false].root_url;
var mode = config["development" || false].title;
var service_title = config["development" || false].title;
var get_folders_and_plots_new_api = function get_folders_and_plots_new_api(params) {
  if (params.plot_search) {
    return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?search=").concat(params.plot_search);
  }

  return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path);
};
var get_folders_and_plots_new_api_with_live_mode = function get_folders_and_plots_new_api_with_live_mode(params) {
  if (params.plot_search) {
    return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?search=").concat(params.plot_search, "&notOlderThan=").concat(params.notOlderThan);
  }

  return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?notOlderThan=").concat(params.notOlderThan);
};
var get_folders_and_plots_old_api = function get_folders_and_plots_old_api(params) {
  if (params.plot_search) {
    return "data/json/archive/".concat(params.run_number).concat(params.dataset_name, "/").concat(params.folders_path, "?search=").concat(params.plot_search);
  }

  return "data/json/archive/".concat(params.run_number).concat(params.dataset_name, "/").concat(params.folders_path);
};
var get_run_list_by_search_old_api = function get_run_list_by_search_old_api(params) {
  return "data/json/samples?match=".concat(params.dataset_name, "&run=").concat(params.run_number);
};
var get_run_list_by_search_new_api = function get_run_list_by_search_new_api(params) {
  return "api/v1/samples?run=".concat(params.run_number, "&lumi=").concat(params.lumi, "&dataset=").concat(params.dataset_name);
};
var get_run_list_by_search_new_api_with_no_older_than = function get_run_list_by_search_new_api_with_no_older_than(params) {
  return "api/v1/samples?run=".concat(params.run_number, "&lumi=").concat(params.lumi, "&dataset=").concat(params.dataset_name, "&notOlderThan=").concat(params.notOlderThan);
};
var get_plot_url = function get_plot_url(params) {
  return "plotfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(params.plot_name, "?").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["get_customize_params"])(params.customizeProps)).concat(params.stats ? '' : 'showstats=0').concat(params.errorBars ? 'showerrbars=1' : '', ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_plot_with_overlay = function get_plot_with_overlay(params) {
  return "plotfairy/overlay?".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["get_customize_params"])(params.customizeProps), "ref=").concat(params.overlay, ";obj=archive/").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name)).concat(params.joined_overlaied_plots_urls, ";").concat(params.stats ? '' : 'showstats=0;').concat(params.errorBars ? 'showerrbars=1;' : '', "norm=").concat(params.normalize, ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_overlaied_plots_urls = function get_overlaied_plots_urls(params) {
  var overlay_plots = params !== null && params !== void 0 && params.overlay_plot && (params === null || params === void 0 ? void 0 : params.overlay_plot.length) > 0 ? params.overlay_plot : [];
  return overlay_plots.map(function (overlay) {
    var dataset_name_overlay = overlay.dataset_name ? overlay.dataset_name : params.dataset_name;
    var label = overlay.label ? overlay.label : overlay.run_number;
    return ";obj=archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(overlay)).concat(dataset_name_overlay).concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), ";reflabel=").concat(label);
  });
};
var get_plot_with_overlay_new_api = function get_plot_with_overlay_new_api(params) {
  var _params$overlaidSepar;

  //empty string in order to set &reflabel= in the start of joined_labels string
  var labels = [''];

  if ((_params$overlaidSepar = params.overlaidSeparately) !== null && _params$overlaidSepar !== void 0 && _params$overlaidSepar.plots) {
    var plots_strings = params.overlaidSeparately.plots.map(function (plot_for_overlay) {
      labels.push(plot_for_overlay.label ? plot_for_overlay.label : params.run_number);
      return "obj=archive/".concat(params.run_number).concat(params.dataset_name, "/").concat(plot_for_overlay.folders_path, "/").concat(encodeURI(plot_for_overlay.plot_name));
    });
    var joined_plots = plots_strings.join('&');
    var joined_labels = labels.join('&reflabel=');
    var norm = params.normalize;
    var stats = params.stats ? '' : 'stats=0';
    var ref = params.overlaidSeparately.ref ? params.overlaidSeparately.ref : 'overlay';
    var error = params.error ? '&showerrbars=1' : '';
    var customization = Object(_utils__WEBPACK_IMPORTED_MODULE_1__["get_customize_params"])(params.customizeProps); //@ts-ignore

    var height = _components_constants__WEBPACK_IMPORTED_MODULE_0__["sizes"][params.size].size.h; //@ts-ignore

    var width = _components_constants__WEBPACK_IMPORTED_MODULE_0__["sizes"][params.size].size.w;
    return "api/v1/render_overlay?obj=archive/".concat(params.run_number).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURI(params.plot_name), "&").concat(joined_plots, "&w=").concat(width, "&h=").concat(height, "&norm=").concat(norm, "&").concat(stats).concat(joined_labels).concat(error, "&").concat(customization, "ref=").concat(ref);
  } else {
    return;
  }
};
var get_jroot_plot = function get_jroot_plot(params) {
  return "jsrootfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), "?jsroot=true;").concat(params.notOlderThan ? "notOlderThan=".concat(params.notOlderThan) : '');
};
var getLumisections = function getLumisections(params) {
  return "api/v1/samples?run=".concat(params.run_number, "&dataset=").concat(params.dataset_name, "&lumi=").concat(params.lumi).concat(functions_config.mode === 'ONLINE' && params.notOlderThan ? "&notOlderThan=".concat(params.notOlderThan) : '');
};
var get_the_latest_runs = function get_the_latest_runs(notOlderThan) {
  return "api/v1/latest_runs?notOlderThan=".concat(notOlderThan);
};

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/process/browser.js */ "./node_modules/process/browser.js"), __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvc3R5bGVkQ29tcG9uZW50cy50cyIsIndlYnBhY2s6Ly9fTl9FLy4vY29uZmlnL2NvbmZpZy50cyJdLCJuYW1lcyI6WyJrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90cyIsImtleWZyYW1lcyIsInRoZW1lIiwiY29sb3JzIiwic2Vjb25kYXJ5IiwibWFpbiIsImNvbW1vbiIsIndoaXRlIiwicHJpbWFyeSIsImxpZ2h0IiwiUGFyZW50V3JhcHBlciIsInN0eWxlZCIsImRpdiIsInByb3BzIiwic2l6ZSIsInciLCJwbG90c0Ftb3VudCIsImgiLCJpc1Bsb3RTZWxlY3RlZCIsImlzTG9hZGluZyIsImFuaW1hdGlvbiIsIkxheW91dE5hbWUiLCJlcnJvciIsIm5vdGlmaWNhdGlvbiIsImJsYWNrIiwiTGF5b3V0V3JhcHBlciIsImF1dG8iLCJQbG90V3JhcHBlciIsInBsb3RTZWxlY3RlZCIsIndpZHRoIiwiaGVpZ2h0IiwiY29uZmlnIiwiZGV2ZWxvcG1lbnQiLCJyb290X3VybCIsInRpdGxlIiwicHJvZHVjdGlvbiIsIm5ld19lbnZfdmFyaWFibGUiLCJwcm9jZXNzIiwibGF5b3V0X2Vudl92YXJpYWJsZSIsImxhdGVzdF9ydW5zX2Vudl92YXJpYWJsZSIsImx1bWlzX2Vudl92YXJpYWJsZSIsImVudiIsIkxVTUlTIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm5ld19iYWNrX2VuZCIsImx1bWlzZWN0aW9uc19vbiIsImxheW91dHMiLCJsYXRlc3RfcnVucyIsIm1vZGUiLCJNT0RFIiwic2VydmljZV90aXRsZSIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpIiwicGFyYW1zIiwicGxvdF9zZWFyY2giLCJnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyIsImRhdGFzZXRfbmFtZSIsImZvbGRlcnNfcGF0aCIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpX3dpdGhfbGl2ZV9tb2RlIiwibm90T2xkZXJUaGFuIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkiLCJydW5fbnVtYmVyIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpIiwibHVtaSIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4iLCJnZXRfcGxvdF91cmwiLCJwbG90X25hbWUiLCJnZXRfY3VzdG9taXplX3BhcmFtcyIsImN1c3RvbWl6ZVByb3BzIiwic3RhdHMiLCJlcnJvckJhcnMiLCJnZXRfcGxvdF93aXRoX292ZXJsYXkiLCJvdmVybGF5IiwiZW5jb2RlVVJJQ29tcG9uZW50Iiwiam9pbmVkX292ZXJsYWllZF9wbG90c191cmxzIiwibm9ybWFsaXplIiwiZ2V0X292ZXJsYWllZF9wbG90c191cmxzIiwib3ZlcmxheV9wbG90cyIsIm92ZXJsYXlfcGxvdCIsImxlbmd0aCIsIm1hcCIsImRhdGFzZXRfbmFtZV9vdmVybGF5IiwibGFiZWwiLCJnZXRfcGxvdF93aXRoX292ZXJsYXlfbmV3X2FwaSIsImxhYmVscyIsIm92ZXJsYWlkU2VwYXJhdGVseSIsInBsb3RzIiwicGxvdHNfc3RyaW5ncyIsInBsb3RfZm9yX292ZXJsYXkiLCJwdXNoIiwiZW5jb2RlVVJJIiwiam9pbmVkX3Bsb3RzIiwiam9pbiIsImpvaW5lZF9sYWJlbHMiLCJub3JtIiwicmVmIiwiY3VzdG9taXphdGlvbiIsInNpemVzIiwiZ2V0X2pyb290X3Bsb3QiLCJnZXRMdW1pc2VjdGlvbnMiLCJnZXRfdGhlX2xhdGVzdF9ydW5zIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUVBO0FBRUEsSUFBTUEsMEJBQTBCLEdBQUdDLG1FQUFILDZEQUVkQyxtREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRlQsRUFHbEJILG1EQUFL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QUMvQyxNQUFEO0FBQUEsc0NBQ1BBLE1BQU0sQ0FBQ1EsVUFEQSxzQkFDc0JSLE1BQU0sQ0FBQ0csWUFEN0IsbUJBRXBCSCxNQUFNLENBQUNXLElBRmEsU0FFTnBCLGdCQUFnQixDQUFDSyxJQUFqQixLQUEwQixRQUExQixJQUFzQ0ksTUFBTSxDQUFDTSxZQUE3QywyQkFDRk4sTUFBTSxDQUFDTSxZQURMLElBRW5CLEVBSnlCO0FBQUEsQ0FBeEI7QUFPQSxJQUFNMEMsbUJBQW1CLEdBQUcsU0FBdEJBLG1CQUFzQixDQUFDMUMsWUFBRCxFQUEwQjtBQUMzRCxtREFBMENBLFlBQTFDO0FBQ0QsQ0FGTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5iMThiZDNlMzU1OGNjOWZmMjBiMS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHN0eWxlZCwgeyBrZXlmcmFtZXMgfSBmcm9tICdzdHlsZWQtY29tcG9uZW50cyc7XHJcbmltcG9ydCB7IFNpemVQcm9wcyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi8uLi8uLi9zdHlsZXMvdGhlbWUnO1xyXG5cclxuY29uc3Qga2V5ZnJhbWVfZm9yX3VwZGF0ZXNfcGxvdHMgPSBrZXlmcmFtZXNgXHJcbiAgMCUge1xyXG4gICAgYmFja2dyb3VuZDogJHt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59O1xyXG4gICAgY29sb3I6ICAke3RoZW1lLmNvbG9ycy5jb21tb24ud2hpdGV9O1xyXG4gIH1cclxuICAxMDAlIHtcclxuICAgIGJhY2tncm91bmQ6ICR7dGhlbWUuY29sb3JzLnByaW1hcnkubGlnaHR9O3NcclxuICB9XHJcbmA7XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IFBhcmVudFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgc2l6ZTogU2l6ZVByb3BzLCBpc0xvYWRpbmc6IHN0cmluZywgYW5pbWF0aW9uOiBzdHJpbmcsIGlzUGxvdFNlbGVjdGVkPzogc3RyaW5nLCBwbG90c0Ftb3VudD86IG51bWJlcjsgfT5gXHJcbiAgICB3aWR0aDogJHsocHJvcHMpID0+IChwcm9wcy5zaXplLncgKyAzMCArIChwcm9wcy5wbG90c0Ftb3VudCA/IHByb3BzLnBsb3RzQW1vdW50IDogNCAqIDQpKX1weDtcclxuICAgIGhlaWdodDogJHsocHJvcHMpID0+IChwcm9wcy5zaXplLmggKyA0MCArIChwcm9wcy5wbG90c0Ftb3VudCA/IHByb3BzLnBsb3RzQW1vdW50IDogNCAqIDQpKX1weDtcclxuICAgIGp1c3RpZnktY29udGVudDogY2VudGVyO1xyXG4gICAgbWFyZ2luOiA0cHg7XHJcbiAgICBiYWNrZ3JvdW5kOiAkeyhwcm9wcykgPT4gcHJvcHMuaXNQbG90U2VsZWN0ZWQgPT09ICd0cnVlJyA/IHRoZW1lLmNvbG9ycy5zZWNvbmRhcnkubGlnaHQgOiB0aGVtZS5jb2xvcnMucHJpbWFyeS5saWdodH07XHJcbiAgICBkaXNwbGF5OiBncmlkO1xyXG4gICAgYWxpZ24taXRlbXM6IGVuZDtcclxuICAgIHBhZGRpbmc6IDhweDtcclxuICAgIGFuaW1hdGlvbi1pdGVyYXRpb24tY291bnQ6IDE7XHJcbiAgICBhbmltYXRpb24tZHVyYXRpb246IDFzO1xyXG4gICAgYW5pbWF0aW9uLW5hbWU6ICR7KHByb3BzKSA9PlxyXG4gICAgcHJvcHMuaXNMb2FkaW5nID09PSAndHJ1ZScgJiYgcHJvcHMuYW5pbWF0aW9uID09PSAndHJ1ZSdcclxuICAgICAgPyBrZXlmcmFtZV9mb3JfdXBkYXRlc19wbG90c1xyXG4gICAgICA6ICcnfTtcclxuYFxyXG5cclxuZXhwb3J0IGNvbnN0IExheW91dE5hbWUgPSBzdHlsZWQuZGl2PHsgZXJyb3I/OiBzdHJpbmcsIGlzUGxvdFNlbGVjdGVkPzogc3RyaW5nIH0+YFxyXG4gICAgcGFkZGluZy1ib3R0b206IDQ7XHJcbiAgICBjb2xvcjogJHtwcm9wcyA9PiBwcm9wcy5lcnJvciA9PT0gJ3RydWUnID8gdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5lcnJvciA6IHRoZW1lLmNvbG9ycy5jb21tb24uYmxhY2t9O1xyXG4gICAgZm9udC13ZWlnaHQ6ICR7cHJvcHMgPT4gcHJvcHMuaXNQbG90U2VsZWN0ZWQgPT09ICd0cnVlJyA/ICdib2xkJyA6ICcnfTtcclxuICAgIHdvcmQtYnJlYWs6IGJyZWFrLXdvcmQ7XHJcbmBcclxuZXhwb3J0IGNvbnN0IExheW91dFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgc2l6ZTogU2l6ZVByb3BzICYgc3RyaW5nLCBhdXRvOiBzdHJpbmcgfT5gXHJcbiAgICAvLyB3aWR0aDogJHsocHJvcHMpID0+IHByb3BzLnNpemUudyA/IGAke3Byb3BzLnNpemUudyArIDEyfXB4YCA6IHByb3BzLnNpemV9O1xyXG4gICAgLy8gaGVpZ2h0OiR7KHByb3BzKSA9PiBwcm9wcy5zaXplLmggPyBgJHtwcm9wcy5zaXplLncgKyAxNn1weGAgOiBwcm9wcy5zaXplfTtcclxuICAgIGRpc3BsYXk6IGdyaWQ7XHJcbiAgICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6ICR7KHByb3BzKSA9PiAocHJvcHMuYXV0byl9O1xyXG4gICAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7XHJcbmA7XHJcblxyXG5leHBvcnQgY29uc3QgUGxvdFdyYXBwZXIgPSBzdHlsZWQuZGl2PHsgcGxvdFNlbGVjdGVkOiBib29sZWFuLCB3aWR0aD86IHN0cmluZywgaGVpZ2h0Pzogc3RyaW5nIH0+YFxyXG4gICAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7XHJcbiAgICBib3JkZXI6ICR7KHByb3BzKSA9PiBwcm9wcy5wbG90U2VsZWN0ZWQgPyBgNHB4IHNvbGlkICR7dGhlbWUuY29sb3JzLnNlY29uZGFyeS5saWdodH1gIDogYDJweCBzb2xpZCAke3RoZW1lLmNvbG9ycy5wcmltYXJ5LmxpZ2h0fWB9O1xyXG4gICAgYWxpZ24taXRlbXM6ICBjZW50ZXIgO1xyXG4gICAgd2lkdGg6ICR7KHByb3BzKSA9PiBwcm9wcy53aWR0aCA/IGBjYWxjKCR7cHJvcHMud2lkdGh9KzhweClgIDogJ2ZpdC1jb250ZW50J307XHJcbiAgICBoZWlnaHQ6ICR7KHByb3BzKSA9PiBwcm9wcy5oZWlnaHQgPyBgY2FsYygke3Byb3BzLmhlaWdodH0rOHB4KWAgOiAnZml0LWNvbnRlbnQnfTs7XHJcbiAgICBjdXJzb3I6ICBwb2ludGVyIDtcclxuICAgIHBhZGRpbmc6IDRweDtcclxuICAgIGFsaWduLXNlbGY6ICBjZW50ZXIgO1xyXG4gICAganVzdGlmeS1zZWxmOiAgYmFzZWxpbmU7XHJcbiAgICBjdXJzb3I6ICR7cHJvcHMgPT4gcHJvcHMucGxvdFNlbGVjdGVkID8gJ3pvb20tb3V0JyA6ICd6b29tLWluJ307XHJcbmAiLCJpbXBvcnQgeyBzaXplcyB9IGZyb20gJy4uL2NvbXBvbmVudHMvY29uc3RhbnRzJztcclxuaW1wb3J0IHsgZ2V0UGF0aE5hbWUgfSBmcm9tICcuLi9jb21wb25lbnRzL3V0aWxzJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBUcmlwbGVQcm9wcyxcclxuICBMdW1pc2VjdGlvblJlcXVlc3RQcm9wcyxcclxufSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IFBhcmFtZXRlcnNGb3JBcGksIFBsb3RQcm9wZXJ0aWVzIH0gZnJvbSAnLi4vcGxvdHNMb2NhbE92ZXJsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IGdldF9jdXN0b21pemVfcGFyYW1zLCBnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyB9IGZyb20gJy4vdXRpbHMnO1xyXG5cclxuY29uc3QgY29uZmlnOiBhbnkgPSB7XHJcbiAgZGV2ZWxvcG1lbnQ6IHtcclxuICAgIHJvb3RfdXJsOiAnaHR0cDovL2xvY2FsaG9zdDo4MDgxLycsXHJcbiAgICB0aXRsZTogJ0RldmVsb3BtZW50JyxcclxuICB9LFxyXG4gIHByb2R1Y3Rpb246IHtcclxuICAgIC8vIHJvb3RfdXJsOiBgaHR0cHM6Ly9kcW0tZ3VpLndlYi5jZXJuLmNoL2FwaS9kcW0vb2ZmbGluZS9gLFxyXG4gICAgcm9vdF91cmw6ICdodHRwOi8vbG9jYWxob3N0OjgwODEvJyxcclxuICAgIC8vIHJvb3RfdXJsOiBgJHtnZXRQYXRoTmFtZSgpfWAsXHJcbiAgICB0aXRsZTogJ09ubGluZS1wbGF5YmFjaycsXHJcbiAgfSxcclxufTtcclxuXHJcbmNvbnN0IG5ld19lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5ORVdfQkFDS19FTkQgPT09ICd0cnVlJztcclxuY29uc3QgbGF5b3V0X2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52LkxBWU9VVFMgPT09ICd0cnVlJztcclxuY29uc3QgbGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTEFURVNUX1JVTlMgPT09ICd0cnVlJztcclxuY29uc3QgbHVtaXNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTFVNSVMgPT09ICd0cnVlJztcclxuXHJcbmV4cG9ydCBjb25zdCBmdW5jdGlvbnNfY29uZmlnOiBhbnkgPSB7XHJcbiAgbmV3X2JhY2tfZW5kOiB7XHJcbiAgICBuZXdfYmFja19lbmQ6IG5ld19lbnZfdmFyaWFibGUgfHwgZmFsc2UsXHJcbiAgICBsdW1pc2VjdGlvbnNfb246IChsdW1pc19lbnZfdmFyaWFibGUgJiYgbmV3X2Vudl92YXJpYWJsZSkgfHwgZmFsc2UsXHJcbiAgICBsYXlvdXRzOiAobGF5b3V0X2Vudl92YXJpYWJsZSAmJiBuZXdfZW52X3ZhcmlhYmxlKSB8fCBmYWxzZSxcclxuICAgIGxhdGVzdF9ydW5zOiAobGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlICYmIG5ld19lbnZfdmFyaWFibGUpIHx8IGZhbHNlLFxyXG4gIH0sXHJcbiAgbW9kZTogcHJvY2Vzcy5lbnYuTU9ERSB8fCAnT0ZGTElORScsXHJcbn07XHJcblxyXG5leHBvcnQgY29uc3Qgcm9vdF91cmwgPSBjb25maWdbcHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50J10ucm9vdF91cmw7XHJcbmV4cG9ydCBjb25zdCBtb2RlID0gY29uZmlnW3Byb2Nlc3MuZW52Lk5PREVfRU5WIHx8ICdkZXZlbG9wbWVudCddLnRpdGxlO1xyXG5cclxuZXhwb3J0IGNvbnN0IHNlcnZpY2VfdGl0bGUgPVxyXG4gIGNvbmZpZ1twcm9jZXNzLmVudi5OT0RFX0VOViB8fCAnZGV2ZWxvcG1lbnQnXS50aXRsZTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgaWYgKHBhcmFtcy5wbG90X3NlYXJjaCkge1xyXG4gICAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9zZWFyY2g9JHtwYXJhbXMucGxvdF9zZWFyY2h9YDtcclxuICB9XHJcbiAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUgPSAoXHJcbiAgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wc1xyXG4pID0+IHtcclxuICBpZiAocGFyYW1zLnBsb3Rfc2VhcmNoKSB7XHJcbiAgICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH0mbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhblxyXG4gICAgICB9YDtcclxuICB9XHJcbiAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/bm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICBpZiAocGFyYW1zLnBsb3Rfc2VhcmNoKSB7XHJcbiAgICByZXR1cm4gYGRhdGEvanNvbi9hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGRhdGEvanNvbi9hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgZGF0YS9qc29uL3NhbXBsZXM/bWF0Y2g9JHtwYXJhbXMuZGF0YXNldF9uYW1lfSZydW49JHtwYXJhbXMucnVuX251bWJlcn1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZsdW1pPSR7cGFyYW1zLmx1bWl9JmRhdGFzZXQ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuID0gKFxyXG4gIHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHNcclxuKSA9PiB7XHJcbiAgcmV0dXJuIGBhcGkvdjEvc2FtcGxlcz9ydW49JHtwYXJhbXMucnVuX251bWJlcn0mbHVtaT0ke3BhcmFtcy5sdW1pfSZkYXRhc2V0PSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0mbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X3Bsb3RfdXJsID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMgJiBQYXJhbWV0ZXJzRm9yQXBpICYgYW55KSA9PiB7XHJcbiAgcmV0dXJuIGBwbG90ZmFpcnkvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHtwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ30/JHtnZXRfY3VzdG9taXplX3BhcmFtcyhcclxuICAgICAgcGFyYW1zLmN1c3RvbWl6ZVByb3BzXHJcbiAgICApfSR7cGFyYW1zLnN0YXRzID8gJycgOiAnc2hvd3N0YXRzPTAnfSR7cGFyYW1zLmVycm9yQmFycyA/ICdzaG93ZXJyYmFycz0xJyA6ICcnXHJcbiAgICB9O3c9JHtwYXJhbXMud2lkdGh9O2g9JHtwYXJhbXMuaGVpZ2h0fWA7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X3Bsb3Rfd2l0aF9vdmVybGF5ID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICByZXR1cm4gYHBsb3RmYWlyeS9vdmVybGF5PyR7Z2V0X2N1c3RvbWl6ZV9wYXJhbXMocGFyYW1zLmN1c3RvbWl6ZVByb3BzKX1yZWY9JHtwYXJhbXMub3ZlcmxheVxyXG4gICAgfTtvYmo9YXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGhcclxuICAgIH0vJHtlbmNvZGVVUklDb21wb25lbnQocGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmcpfSR7cGFyYW1zLmpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJsc1xyXG4gICAgfTske3BhcmFtcy5zdGF0cyA/ICcnIDogJ3Nob3dzdGF0cz0wOyd9JHtwYXJhbXMuZXJyb3JCYXJzID8gJ3Nob3dlcnJiYXJzPTE7JyA6ICcnXHJcbiAgICB9bm9ybT0ke3BhcmFtcy5ub3JtYWxpemV9O3c9JHtwYXJhbXMud2lkdGh9O2g9JHtwYXJhbXMuaGVpZ2h0fWA7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X292ZXJsYWllZF9wbG90c191cmxzID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICBjb25zdCBvdmVybGF5X3Bsb3RzID1cclxuICAgIHBhcmFtcz8ub3ZlcmxheV9wbG90ICYmIHBhcmFtcz8ub3ZlcmxheV9wbG90Lmxlbmd0aCA+IDBcclxuICAgICAgPyBwYXJhbXMub3ZlcmxheV9wbG90XHJcbiAgICAgIDogW107XHJcblxyXG4gIHJldHVybiBvdmVybGF5X3Bsb3RzLm1hcCgob3ZlcmxheTogVHJpcGxlUHJvcHMpID0+IHtcclxuICAgIGNvbnN0IGRhdGFzZXRfbmFtZV9vdmVybGF5ID0gb3ZlcmxheS5kYXRhc2V0X25hbWVcclxuICAgICAgPyBvdmVybGF5LmRhdGFzZXRfbmFtZVxyXG4gICAgICA6IHBhcmFtcy5kYXRhc2V0X25hbWU7XHJcbiAgICBjb25zdCBsYWJlbCA9IG92ZXJsYXkubGFiZWwgPyBvdmVybGF5LmxhYmVsIDogb3ZlcmxheS5ydW5fbnVtYmVyO1xyXG4gICAgcmV0dXJuIGA7b2JqPWFyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhcclxuICAgICAgb3ZlcmxheVxyXG4gICAgKX0ke2RhdGFzZXRfbmFtZV9vdmVybGF5fSR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHtlbmNvZGVVUklDb21wb25lbnQoXHJcbiAgICAgIHBhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nXHJcbiAgICApfTtyZWZsYWJlbD0ke2xhYmVsfWA7XHJcbiAgfSk7XHJcbn07XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9wbG90X3dpdGhfb3ZlcmxheV9uZXdfYXBpID0gKHBhcmFtczogUGFyYW1ldGVyc0ZvckFwaSkgPT4ge1xyXG4gIC8vZW1wdHkgc3RyaW5nIGluIG9yZGVyIHRvIHNldCAmcmVmbGFiZWw9IGluIHRoZSBzdGFydCBvZiBqb2luZWRfbGFiZWxzIHN0cmluZ1xyXG4gIGNvbnN0IGxhYmVsczogc3RyaW5nW10gPSBbJyddXHJcbiAgaWYgKHBhcmFtcy5vdmVybGFpZFNlcGFyYXRlbHk/LnBsb3RzKSB7XHJcbiAgICBjb25zdCBwbG90c19zdHJpbmdzID0gcGFyYW1zLm92ZXJsYWlkU2VwYXJhdGVseS5wbG90cy5tYXAoKHBsb3RfZm9yX292ZXJsYXk6IFBsb3RQcm9wZXJ0aWVzKSA9PiB7XHJcbiAgICAgIGxhYmVscy5wdXNoKHBsb3RfZm9yX292ZXJsYXkubGFiZWwgPyBwbG90X2Zvcl9vdmVybGF5LmxhYmVsIDogcGFyYW1zLnJ1bl9udW1iZXIpXHJcbiAgICAgIHJldHVybiAoYG9iaj1hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3Bsb3RfZm9yX292ZXJsYXkuZm9sZGVyc19wYXRofS8keyhlbmNvZGVVUkkocGxvdF9mb3Jfb3ZlcmxheS5wbG90X25hbWUpKX1gKVxyXG4gICAgfSlcclxuICAgIGNvbnN0IGpvaW5lZF9wbG90cyA9IHBsb3RzX3N0cmluZ3Muam9pbignJicpXHJcbiAgICBjb25zdCBqb2luZWRfbGFiZWxzID0gbGFiZWxzLmpvaW4oJyZyZWZsYWJlbD0nKVxyXG4gICAgY29uc3Qgbm9ybSA9IHBhcmFtcy5ub3JtYWxpemVcclxuICAgIGNvbnN0IHN0YXRzID0gcGFyYW1zLnN0YXRzID8gJycgOiAnc3RhdHM9MCdcclxuICAgIGNvbnN0IHJlZiA9IHBhcmFtcy5vdmVybGFpZFNlcGFyYXRlbHkucmVmID8gcGFyYW1zLm92ZXJsYWlkU2VwYXJhdGVseS5yZWYgOiAnb3ZlcmxheSdcclxuICAgIGNvbnN0IGVycm9yID0gcGFyYW1zLmVycm9yID8gJyZzaG93ZXJyYmFycz0xJyA6ICcnXHJcbiAgICBjb25zdCBjdXN0b21pemF0aW9uID0gZ2V0X2N1c3RvbWl6ZV9wYXJhbXMocGFyYW1zLmN1c3RvbWl6ZVByb3BzKVxyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBjb25zdCBoZWlnaHQgPSBzaXplc1twYXJhbXMuc2l6ZV0uc2l6ZS5oXHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIGNvbnN0IHdpZHRoID0gc2l6ZXNbcGFyYW1zLnNpemVdLnNpemUud1xyXG5cclxuICAgIHJldHVybiBgYXBpL3YxL3JlbmRlcl9vdmVybGF5P29iaj1hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7KGVuY29kZVVSSShwYXJhbXMucGxvdF9uYW1lKSl9JiR7am9pbmVkX3Bsb3RzfSZ3PSR7d2lkdGh9Jmg9JHtoZWlnaHR9Jm5vcm09JHtub3JtfSYke3N0YXRzfSR7am9pbmVkX2xhYmVsc30ke2Vycm9yfSYke2N1c3RvbWl6YXRpb259cmVmPSR7cmVmfWBcclxuICB9XHJcbiAgZWxzZSB7XHJcbiAgICByZXR1cm5cclxuICB9XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfanJvb3RfcGxvdCA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PlxyXG4gIGBqc3Jvb3RmYWlyeS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHtlbmNvZGVVUklDb21wb25lbnQoXHJcbiAgICBwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ1xyXG4gICl9P2pzcm9vdD10cnVlOyR7cGFyYW1zLm5vdE9sZGVyVGhhbiA/IGBub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWAgOiAnJ31gO1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldEx1bWlzZWN0aW9ucyA9IChwYXJhbXM6IEx1bWlzZWN0aW9uUmVxdWVzdFByb3BzKSA9PlxyXG4gIGBhcGkvdjEvc2FtcGxlcz9ydW49JHtwYXJhbXMucnVuX251bWJlcn0mZGF0YXNldD0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICB9Jmx1bWk9JHtwYXJhbXMubHVtaX0ke2Z1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScgJiYgcGFyYW1zLm5vdE9sZGVyVGhhblxyXG4gICAgPyBgJm5vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW59YFxyXG4gICAgOiAnJ1xyXG4gIH1gO1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF90aGVfbGF0ZXN0X3J1bnMgPSAobm90T2xkZXJUaGFuOiBudW1iZXIpID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9sYXRlc3RfcnVucz9ub3RPbGRlclRoYW49JHtub3RPbGRlclRoYW59YDtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
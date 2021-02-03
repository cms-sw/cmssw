webpackHotUpdate_N_E("pages/index",{

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
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../components/utils */ "./components/utils.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils */ "./config/utils.ts");



var config = {
  development: {
    root_url: 'http://localhost:8086/',
    title: 'Development'
  },
  production: {
    // root_url: `https://dqm-gui.web.cern.ch/api/dqm/offline/`,
    // root_url: 'http://localhost:8081/',
    root_url: "".concat(Object(_components_utils__WEBPACK_IMPORTED_MODULE_1__["getPathName"])()),
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
    return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?search=").concat(params.plot_search);
  }

  return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path);
};
var get_folders_and_plots_new_api_with_live_mode = function get_folders_and_plots_new_api_with_live_mode(params) {
  if (params.plot_search) {
    return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?search=").concat(params.plot_search, "&notOlderThan=").concat(params.notOlderThan);
  }

  return "api/v1/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "?notOlderThan=").concat(params.notOlderThan);
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
  return "plotfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(params.plot_name, "?").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["get_customize_params"])(params.customizeProps)).concat(params.stats ? '' : 'showstats=0').concat(params.errorBars ? 'showerrbars=1' : '', ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_plot_with_overlay = function get_plot_with_overlay(params) {
  return "plotfairy/overlay?".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["get_customize_params"])(params.customizeProps), "ref=").concat(params.overlay, ";obj=archive/").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name)).concat(params.joined_overlaied_plots_urls, ";").concat(params.stats ? '' : 'showstats=0;').concat(params.errorBars ? 'showerrbars=1;' : '', "norm=").concat(params.normalize, ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_overlaied_plots_urls = function get_overlaied_plots_urls(params) {
  var overlay_plots = params !== null && params !== void 0 && params.overlay_plot && (params === null || params === void 0 ? void 0 : params.overlay_plot.length) > 0 ? params.overlay_plot : [];
  return overlay_plots.map(function (overlay) {
    var dataset_name_overlay = overlay.dataset_name ? overlay.dataset_name : params.dataset_name;
    var label = overlay.label ? overlay.label : overlay.run_number;
    return ";obj=archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(overlay)).concat(dataset_name_overlay).concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), ";reflabel=").concat(label);
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
    var customization = Object(_utils__WEBPACK_IMPORTED_MODULE_2__["get_customize_params"])(params.customizeProps); //@ts-ignore

    var height = _components_constants__WEBPACK_IMPORTED_MODULE_0__["sizes"][params.size].size.h; //@ts-ignore

    var width = _components_constants__WEBPACK_IMPORTED_MODULE_0__["sizes"][params.size].size.w;
    return "api/v1/render_overlay?obj=archive/".concat(params.run_number).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURI(params.plot_name), "&").concat(joined_plots, "&w=").concat(width, "&h=").concat(height, "&norm=").concat(norm, "&").concat(stats).concat(joined_labels).concat(error, "&").concat(customization, "ref=").concat(ref);
  } else {
    return;
  }
};
var get_jroot_plot = function get_jroot_plot(params) {
  return "jsrootfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_2__["getRunsWithLumisections"])(params)).concat(params.dataset_name, "/").concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), "?jsroot=true;").concat(params.notOlderThan ? "notOlderThan=".concat(params.notOlderThan) : '');
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29uZmlnL2NvbmZpZy50cyJdLCJuYW1lcyI6WyJjb25maWciLCJkZXZlbG9wbWVudCIsInJvb3RfdXJsIiwidGl0bGUiLCJwcm9kdWN0aW9uIiwiZ2V0UGF0aE5hbWUiLCJuZXdfZW52X3ZhcmlhYmxlIiwicHJvY2VzcyIsImxheW91dF9lbnZfdmFyaWFibGUiLCJsYXRlc3RfcnVuc19lbnZfdmFyaWFibGUiLCJsdW1pc19lbnZfdmFyaWFibGUiLCJlbnYiLCJMVU1JUyIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJsdW1pc2VjdGlvbnNfb24iLCJsYXlvdXRzIiwibGF0ZXN0X3J1bnMiLCJtb2RlIiwiTU9ERSIsInNlcnZpY2VfdGl0bGUiLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSIsInBhcmFtcyIsInBsb3Rfc2VhcmNoIiwiZ2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMiLCJkYXRhc2V0X25hbWUiLCJmb2xkZXJzX3BhdGgiLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZSIsIm5vdE9sZGVyVGhhbiIsImdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpIiwicnVuX251bWJlciIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfb2xkX2FwaSIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSIsImx1bWkiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuIiwiZ2V0X3Bsb3RfdXJsIiwicGxvdF9uYW1lIiwiZ2V0X2N1c3RvbWl6ZV9wYXJhbXMiLCJjdXN0b21pemVQcm9wcyIsInN0YXRzIiwiZXJyb3JCYXJzIiwid2lkdGgiLCJoZWlnaHQiLCJnZXRfcGxvdF93aXRoX292ZXJsYXkiLCJvdmVybGF5IiwiZW5jb2RlVVJJQ29tcG9uZW50Iiwiam9pbmVkX292ZXJsYWllZF9wbG90c191cmxzIiwibm9ybWFsaXplIiwiZ2V0X292ZXJsYWllZF9wbG90c191cmxzIiwib3ZlcmxheV9wbG90cyIsIm92ZXJsYXlfcGxvdCIsImxlbmd0aCIsIm1hcCIsImRhdGFzZXRfbmFtZV9vdmVybGF5IiwibGFiZWwiLCJnZXRfcGxvdF93aXRoX292ZXJsYXlfbmV3X2FwaSIsImxhYmVscyIsIm92ZXJsYWlkU2VwYXJhdGVseSIsInBsb3RzIiwicGxvdHNfc3RyaW5ncyIsInBsb3RfZm9yX292ZXJsYXkiLCJwdXNoIiwiZW5jb2RlVVJJIiwiam9pbmVkX3Bsb3RzIiwiam9pbiIsImpvaW5lZF9sYWJlbHMiLCJub3JtIiwicmVmIiwiZXJyb3IiLCJjdXN0b21pemF0aW9uIiwic2l6ZXMiLCJzaXplIiwiaCIsInciLCJnZXRfanJvb3RfcGxvdCIsImdldEx1bWlzZWN0aW9ucyIsImdldF90aGVfbGF0ZXN0X3J1bnMiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBT0E7QUFFQSxJQUFNQSxNQUFXLEdBQUc7QUFDbEJDLGFBQVcsRUFBRTtBQUNYQyxZQUFRLEVBQUUsd0JBREM7QUFFWEMsU0FBSyxFQUFFO0FBRkksR0FESztBQUtsQkMsWUFBVSxFQUFFO0FBQ1Y7QUFDQTtBQUNBRixZQUFRLFlBQUtHLHFFQUFXLEVBQWhCLENBSEU7QUFJVkYsU0FBSyxFQUFFO0FBSkc7QUFMTSxDQUFwQjtBQWFBLElBQU1HLGdCQUFnQixHQUFHQyxNQUFBLEtBQTZCLE1BQXREO0FBQ0EsSUFBTUMsbUJBQW1CLEdBQUdELE1BQUEsS0FBd0IsTUFBcEQ7QUFDQSxJQUFNRSx3QkFBd0IsR0FBR0YsTUFBQSxLQUE0QixNQUE3RDtBQUNBLElBQU1HLGtCQUFrQixHQUFHSCxPQUFPLENBQUNJLEdBQVIsQ0FBWUMsS0FBWixLQUFzQixNQUFqRDtBQUVPLElB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WUFBdkUsY0FBdUZILE1BQU0sQ0FBQ0ksWUFBOUYsY0FBK0dvQyxTQUFTLENBQUN4QyxNQUFNLENBQUNjLFNBQVIsQ0FBeEgsY0FBK0kyQixZQUEvSSxnQkFBaUt0QixLQUFqSyxnQkFBNEtDLE1BQTVLLG1CQUEyTHdCLElBQTNMLGNBQW1NM0IsS0FBbk0sU0FBMk0wQixhQUEzTSxTQUEyTkcsS0FBM04sY0FBb09DLGFBQXBPLGlCQUF3UEYsR0FBeFA7QUFDRCxHQWxCRCxNQW1CSztBQUNIO0FBQ0Q7QUFDRixDQXpCTTtBQTJCQSxJQUFNTyxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLENBQUNwRCxNQUFEO0FBQUEsdUNBQ0xFLHNFQUF1QixDQUFDRixNQUFELENBRGxCLFNBQzZCQSxNQUFNLENBQUNHLFlBRHBDLGNBRXhCSCxNQUFNLENBQUNJLFlBRmlCLGNBRURtQixrQkFBa0IsQ0FDM0N2QixNQUFNLENBQUNjLFNBRG9DLENBRmpCLDBCQUlYZCxNQUFNLENBQUNNLFlBQVAsMEJBQXNDTixNQUFNLENBQUNNLFlBQTdDLElBQThELEVBSm5EO0FBQUEsQ0FBdkI7QUFNQSxJQUFNK0MsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixDQUFDckQsTUFBRDtBQUFBLHNDQUNQQSxNQUFNLENBQUNRLFVBREEsc0JBQ3NCUixNQUFNLENBQUNHLFlBRDdCLG1CQUVwQkgsTUFBTSxDQUFDVyxJQUZhLFNBRU5wQixnQkFBZ0IsQ0FBQ0ssSUFBakIsS0FBMEIsUUFBMUIsSUFBc0NJLE1BQU0sQ0FBQ00sWUFBN0MsMkJBQ0ZOLE1BQU0sQ0FBQ00sWUFETCxJQUVuQixFQUp5QjtBQUFBLENBQXhCO0FBT0EsSUFBTWdELG1CQUFtQixHQUFHLFNBQXRCQSxtQkFBc0IsQ0FBQ2hELFlBQUQsRUFBMEI7QUFDM0QsbURBQTBDQSxZQUExQztBQUNELENBRk0iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDAwZWVhODNmY2VmZWY1YjU2OWQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IHNpemVzIH0gZnJvbSAnLi4vY29tcG9uZW50cy9jb25zdGFudHMnO1xyXG5pbXBvcnQgeyBnZXRQYXRoTmFtZSB9IGZyb20gJy4uL2NvbXBvbmVudHMvdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFRyaXBsZVByb3BzLFxyXG4gIEx1bWlzZWN0aW9uUmVxdWVzdFByb3BzLFxyXG59IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgUGFyYW1ldGVyc0ZvckFwaSwgUGxvdFByb3BlcnRpZXMgfSBmcm9tICcuLi9wbG90c0xvY2FsT3ZlcmxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgZ2V0X2N1c3RvbWl6ZV9wYXJhbXMsIGdldFJ1bnNXaXRoTHVtaXNlY3Rpb25zIH0gZnJvbSAnLi91dGlscyc7XHJcblxyXG5jb25zdCBjb25maWc6IGFueSA9IHtcclxuICBkZXZlbG9wbWVudDoge1xyXG4gICAgcm9vdF91cmw6ICdodHRwOi8vbG9jYWxob3N0OjgwODYvJyxcclxuICAgIHRpdGxlOiAnRGV2ZWxvcG1lbnQnLFxyXG4gIH0sXHJcbiAgcHJvZHVjdGlvbjoge1xyXG4gICAgLy8gcm9vdF91cmw6IGBodHRwczovL2RxbS1ndWkud2ViLmNlcm4uY2gvYXBpL2RxbS9vZmZsaW5lL2AsXHJcbiAgICAvLyByb290X3VybDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4MS8nLFxyXG4gICAgcm9vdF91cmw6IGAke2dldFBhdGhOYW1lKCl9YCxcclxuICAgIHRpdGxlOiAnT25saW5lLXBsYXliYWNrJyxcclxuICB9LFxyXG59O1xyXG5cclxuY29uc3QgbmV3X2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52Lk5FV19CQUNLX0VORCA9PT0gJ3RydWUnO1xyXG5jb25zdCBsYXlvdXRfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTEFZT1VUUyA9PT0gJ3RydWUnO1xyXG5jb25zdCBsYXRlc3RfcnVuc19lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5MQVRFU1RfUlVOUyA9PT0gJ3RydWUnO1xyXG5jb25zdCBsdW1pc19lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5MVU1JUyA9PT0gJ3RydWUnO1xyXG5cclxuZXhwb3J0IGNvbnN0IGZ1bmN0aW9uc19jb25maWc6IGFueSA9IHtcclxuICBuZXdfYmFja19lbmQ6IHtcclxuICAgIG5ld19iYWNrX2VuZDogbmV3X2Vudl92YXJpYWJsZSB8fCBmYWxzZSxcclxuICAgIGx1bWlzZWN0aW9uc19vbjogKGx1bWlzX2Vudl92YXJpYWJsZSAmJiBuZXdfZW52X3ZhcmlhYmxlKSB8fCBmYWxzZSxcclxuICAgIGxheW91dHM6IChsYXlvdXRfZW52X3ZhcmlhYmxlICYmIG5ld19lbnZfdmFyaWFibGUpIHx8IGZhbHNlLFxyXG4gICAgbGF0ZXN0X3J1bnM6IChsYXRlc3RfcnVuc19lbnZfdmFyaWFibGUgJiYgbmV3X2Vudl92YXJpYWJsZSkgfHwgZmFsc2UsXHJcbiAgfSxcclxuICBtb2RlOiBwcm9jZXNzLmVudi5NT0RFIHx8ICdPRkZMSU5FJyxcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCByb290X3VybCA9IGNvbmZpZ1twcm9jZXNzLmVudi5OT0RFX0VOViB8fCAnZGV2ZWxvcG1lbnQnXS5yb290X3VybDtcclxuZXhwb3J0IGNvbnN0IG1vZGUgPSBjb25maWdbcHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50J10udGl0bGU7XHJcblxyXG5leHBvcnQgY29uc3Qgc2VydmljZV90aXRsZSA9XHJcbiAgY29uZmlnW3Byb2Nlc3MuZW52Lk5PREVfRU5WIHx8ICdkZXZlbG9wbWVudCddLnRpdGxlO1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICBpZiAocGFyYW1zLnBsb3Rfc2VhcmNoKSB7XHJcbiAgICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZSA9IChcclxuICBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzXHJcbikgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/c2VhcmNoPSR7cGFyYW1zLnBsb3Rfc2VhcmNofSZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICAgIH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9ub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWA7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgZGF0YS9qc29uL2FyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/c2VhcmNoPSR7cGFyYW1zLnBsb3Rfc2VhcmNofWA7XHJcbiAgfVxyXG4gIHJldHVybiBgZGF0YS9qc29uL2FyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfb2xkX2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgcmV0dXJuIGBkYXRhL2pzb24vc2FtcGxlcz9tYXRjaD0ke3BhcmFtcy5kYXRhc2V0X25hbWV9JnJ1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgYXBpL3YxL3NhbXBsZXM/cnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9Jmx1bWk9JHtwYXJhbXMubHVtaX0mZGF0YXNldD0ke3BhcmFtcy5kYXRhc2V0X25hbWV9YDtcclxufTtcclxuZXhwb3J0IGNvbnN0IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4gPSAoXHJcbiAgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wc1xyXG4pID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZsdW1pPSR7cGFyYW1zLmx1bWl9JmRhdGFzZXQ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfSZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF91cmwgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcyAmIFBhcmFtZXRlcnNGb3JBcGkgJiBhbnkpID0+IHtcclxuICByZXR1cm4gYHBsb3RmYWlyeS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofS8ke3BhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nfT8ke2dldF9jdXN0b21pemVfcGFyYW1zKFxyXG4gICAgICBwYXJhbXMuY3VzdG9taXplUHJvcHNcclxuICAgICl9JHtwYXJhbXMuc3RhdHMgPyAnJyA6ICdzaG93c3RhdHM9MCd9JHtwYXJhbXMuZXJyb3JCYXJzID8gJ3Nob3dlcnJiYXJzPTEnIDogJydcclxuICAgIH07dz0ke3BhcmFtcy53aWR0aH07aD0ke3BhcmFtcy5oZWlnaHR9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF93aXRoX292ZXJsYXkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgcGxvdGZhaXJ5L292ZXJsYXk/JHtnZXRfY3VzdG9taXplX3BhcmFtcyhwYXJhbXMuY3VzdG9taXplUHJvcHMpfXJlZj0ke3BhcmFtcy5vdmVybGF5XHJcbiAgICB9O29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aFxyXG4gICAgfS8ke2VuY29kZVVSSUNvbXBvbmVudChwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZyl9JHtwYXJhbXMuam9pbmVkX292ZXJsYWllZF9wbG90c191cmxzXHJcbiAgICB9OyR7cGFyYW1zLnN0YXRzID8gJycgOiAnc2hvd3N0YXRzPTA7J30ke3BhcmFtcy5lcnJvckJhcnMgPyAnc2hvd2VycmJhcnM9MTsnIDogJydcclxuICAgIH1ub3JtPSR7cGFyYW1zLm5vcm1hbGl6ZX07dz0ke3BhcmFtcy53aWR0aH07aD0ke3BhcmFtcy5oZWlnaHR9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIGNvbnN0IG92ZXJsYXlfcGxvdHMgPVxyXG4gICAgcGFyYW1zPy5vdmVybGF5X3Bsb3QgJiYgcGFyYW1zPy5vdmVybGF5X3Bsb3QubGVuZ3RoID4gMFxyXG4gICAgICA/IHBhcmFtcy5vdmVybGF5X3Bsb3RcclxuICAgICAgOiBbXTtcclxuXHJcbiAgcmV0dXJuIG92ZXJsYXlfcGxvdHMubWFwKChvdmVybGF5OiBUcmlwbGVQcm9wcykgPT4ge1xyXG4gICAgY29uc3QgZGF0YXNldF9uYW1lX292ZXJsYXkgPSBvdmVybGF5LmRhdGFzZXRfbmFtZVxyXG4gICAgICA/IG92ZXJsYXkuZGF0YXNldF9uYW1lXHJcbiAgICAgIDogcGFyYW1zLmRhdGFzZXRfbmFtZTtcclxuICAgIGNvbnN0IGxhYmVsID0gb3ZlcmxheS5sYWJlbCA/IG92ZXJsYXkubGFiZWwgOiBvdmVybGF5LnJ1bl9udW1iZXI7XHJcbiAgICByZXR1cm4gYDtvYmo9YXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKFxyXG4gICAgICBvdmVybGF5XHJcbiAgICApfSR7ZGF0YXNldF9uYW1lX292ZXJsYXl9JHtwYXJhbXMuZm9sZGVyc19wYXRofS8ke2VuY29kZVVSSUNvbXBvbmVudChcclxuICAgICAgcGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmdcclxuICAgICl9O3JlZmxhYmVsPSR7bGFiZWx9YDtcclxuICB9KTtcclxufTtcclxuXHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X3Bsb3Rfd2l0aF9vdmVybGF5X25ld19hcGkgPSAocGFyYW1zOiBQYXJhbWV0ZXJzRm9yQXBpKSA9PiB7XHJcbiAgLy9lbXB0eSBzdHJpbmcgaW4gb3JkZXIgdG8gc2V0ICZyZWZsYWJlbD0gaW4gdGhlIHN0YXJ0IG9mIGpvaW5lZF9sYWJlbHMgc3RyaW5nXHJcbiAgY29uc3QgbGFiZWxzOiBzdHJpbmdbXSA9IFsnJ11cclxuICBpZiAocGFyYW1zLm92ZXJsYWlkU2VwYXJhdGVseT8ucGxvdHMpIHtcclxuICAgIGNvbnN0IHBsb3RzX3N0cmluZ3MgPSBwYXJhbXMub3ZlcmxhaWRTZXBhcmF0ZWx5LnBsb3RzLm1hcCgocGxvdF9mb3Jfb3ZlcmxheTogUGxvdFByb3BlcnRpZXMpID0+IHtcclxuICAgICAgbGFiZWxzLnB1c2gocGxvdF9mb3Jfb3ZlcmxheS5sYWJlbCA/IHBsb3RfZm9yX292ZXJsYXkubGFiZWwgOiBwYXJhbXMucnVuX251bWJlcilcclxuICAgICAgcmV0dXJuIChgb2JqPWFyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGxvdF9mb3Jfb3ZlcmxheS5mb2xkZXJzX3BhdGh9LyR7KGVuY29kZVVSSShwbG90X2Zvcl9vdmVybGF5LnBsb3RfbmFtZSkpfWApXHJcbiAgICB9KVxyXG4gICAgY29uc3Qgam9pbmVkX3Bsb3RzID0gcGxvdHNfc3RyaW5ncy5qb2luKCcmJylcclxuICAgIGNvbnN0IGpvaW5lZF9sYWJlbHMgPSBsYWJlbHMuam9pbignJnJlZmxhYmVsPScpXHJcbiAgICBjb25zdCBub3JtID0gcGFyYW1zLm5vcm1hbGl6ZVxyXG4gICAgY29uc3Qgc3RhdHMgPSBwYXJhbXMuc3RhdHMgPyAnJyA6ICdzdGF0cz0wJ1xyXG4gICAgY29uc3QgcmVmID0gcGFyYW1zLm92ZXJsYWlkU2VwYXJhdGVseS5yZWYgPyBwYXJhbXMub3ZlcmxhaWRTZXBhcmF0ZWx5LnJlZiA6ICdvdmVybGF5J1xyXG4gICAgY29uc3QgZXJyb3IgPSBwYXJhbXMuZXJyb3IgPyAnJnNob3dlcnJiYXJzPTEnIDogJydcclxuICAgIGNvbnN0IGN1c3RvbWl6YXRpb24gPSBnZXRfY3VzdG9taXplX3BhcmFtcyhwYXJhbXMuY3VzdG9taXplUHJvcHMpXHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIGNvbnN0IGhlaWdodCA9IHNpemVzW3BhcmFtcy5zaXplXS5zaXplLmhcclxuICAgIC8vQHRzLWlnbm9yZVxyXG4gICAgY29uc3Qgd2lkdGggPSBzaXplc1twYXJhbXMuc2l6ZV0uc2l6ZS53XHJcblxyXG4gICAgcmV0dXJuIGBhcGkvdjEvcmVuZGVyX292ZXJsYXk/b2JqPWFyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHsoZW5jb2RlVVJJKHBhcmFtcy5wbG90X25hbWUpKX0mJHtqb2luZWRfcGxvdHN9Jnc9JHt3aWR0aH0maD0ke2hlaWdodH0mbm9ybT0ke25vcm19JiR7c3RhdHN9JHtqb2luZWRfbGFiZWxzfSR7ZXJyb3J9JiR7Y3VzdG9taXphdGlvbn1yZWY9JHtyZWZ9YFxyXG4gIH1cclxuICBlbHNlIHtcclxuICAgIHJldHVyblxyXG4gIH1cclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9qcm9vdF9wbG90ID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+XHJcbiAgYGpzcm9vdGZhaXJ5L2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofS8ke2VuY29kZVVSSUNvbXBvbmVudChcclxuICAgIHBhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nXHJcbiAgKX0/anNyb290PXRydWU7JHtwYXJhbXMubm90T2xkZXJUaGFuID8gYG5vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW59YCA6ICcnfWA7XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0THVtaXNlY3Rpb25zID0gKHBhcmFtczogTHVtaXNlY3Rpb25SZXF1ZXN0UHJvcHMpID0+XHJcbiAgYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZkYXRhc2V0PSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0mbHVtaT0ke3BhcmFtcy5sdW1pfSR7ZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJyAmJiBwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICA/IGAmbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gXHJcbiAgICA6ICcnXHJcbiAgfWA7XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X3RoZV9sYXRlc3RfcnVucyA9IChub3RPbGRlclRoYW46IG51bWJlcikgPT4ge1xyXG4gIHJldHVybiBgYXBpL3YxL2xhdGVzdF9ydW5zP25vdE9sZGVyVGhhbj0ke25vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
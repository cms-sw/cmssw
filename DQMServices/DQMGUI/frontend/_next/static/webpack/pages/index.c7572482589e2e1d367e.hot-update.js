webpackHotUpdate_N_E("pages/index",{

/***/ "./config/config.ts":
/*!**************************!*\
  !*** ./config/config.ts ***!
  \**************************/
/*! exports provided: functions_config, root_url, service_title, get_folders_and_plots_new_api, get_folders_and_plots_new_api_with_live_mode, get_folders_and_plots_old_api, get_run_list_by_search_old_api, get_run_list_by_search_new_api, get_run_list_by_search_new_api_with_no_older_than, get_plot_url, get_plot_with_overlay, get_overlaied_plots_urls, get_jroot_plot, getLumisections, get_the_latest_runs */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(process, module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "functions_config", function() { return functions_config; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "root_url", function() { return root_url; });
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
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_jroot_plot", function() { return get_jroot_plot; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getLumisections", function() { return getLumisections; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_the_latest_runs", function() { return get_the_latest_runs; });
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../components/utils */ "./components/utils.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils */ "./config/utils.ts");


var config = {
  development: {
    root_url: 'http://localhost:8086',
    title: 'Development'
  },
  production: {
    root_url: Object(_components_utils__WEBPACK_IMPORTED_MODULE_0__["getPathName"])(),
    // root_url: 'https://dqm-gui.web.cern.ch/api/dqm/offline/',
    // root_url: './',
    // root_url: 'http://localhost:8086/',
    title: 'Offline'
  }
};
var new_env_variable = "true" === 'true';
var layout_env_variable = "true" === 'true';
var latest_runs_env_variable = "true" === 'true';
var lumis_env_variable = "true" === 'true';
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
  return "plotfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name).concat(params.folders_path, "/").concat(params.plot_name, "?").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["get_customize_params"])(params.customizeProps)).concat(params.stats ? '' : 'showstats=0;').concat(params.errorBars ? 'showerrbars=1;' : '', ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_plot_with_overlay = function get_plot_with_overlay(params) {
  return "plotfairy/overlay?".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["get_customize_params"])(params.customizeProps), "ref=").concat(params.overlay, ";obj=archive/").concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name).concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name)).concat(params.joined_overlaied_plots_urls, ";").concat(params.stats ? '' : 'showstats=0;').concat(params.errorBars ? 'showerrbars=1;' : '', "norm=").concat(params.normalize, ";w=").concat(params.width, ";h=").concat(params.height);
};
var get_overlaied_plots_urls = function get_overlaied_plots_urls(params) {
  var overlay_plots = (params === null || params === void 0 ? void 0 : params.overlay_plot) && (params === null || params === void 0 ? void 0 : params.overlay_plot.length) > 0 ? params.overlay_plot : [];
  return overlay_plots.map(function (overlay) {
    var dataset_name_overlay = overlay.dataset_name ? overlay.dataset_name : params.dataset_name;
    var label = overlay.label ? overlay.label : overlay.run_number;
    return ";obj=archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(overlay)).concat(dataset_name_overlay).concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), ";reflabel=").concat(label);
  });
};
var get_jroot_plot = function get_jroot_plot(params) {
  return "jsrootfairy/archive/".concat(Object(_utils__WEBPACK_IMPORTED_MODULE_1__["getRunsWithLumisections"])(params)).concat(params.dataset_name).concat(params.folders_path, "/").concat(encodeURIComponent(params.plot_name), "?jsroot=true;notOlderThan=").concat(params.notOlderThan);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29uZmlnL2NvbmZpZy50cyJdLCJuYW1lcyI6WyJjb25maWciLCJkZXZlbG9wbWVudCIsInJvb3RfdXJsIiwidGl0bGUiLCJwcm9kdWN0aW9uIiwiZ2V0UGF0aE5hbWUiLCJuZXdfZW52X3ZhcmlhYmxlIiwicHJvY2VzcyIsImxheW91dF9lbnZfdmFyaWFibGUiLCJsYXRlc3RfcnVuc19lbnZfdmFyaWFibGUiLCJsdW1pc19lbnZfdmFyaWFibGUiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIiwibGF5b3V0cyIsImxhdGVzdF9ydW5zIiwibW9kZSIsImVudiIsIk1PREUiLCJzZXJ2aWNlX3RpdGxlIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGkiLCJwYXJhbXMiLCJwbG90X3NlYXJjaCIsImdldFJ1bnNXaXRoTHVtaXNlY3Rpb25zIiwiZGF0YXNldF9uYW1lIiwiZm9sZGVyc19wYXRoIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUiLCJub3RPbGRlclRoYW4iLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfb2xkX2FwaSIsInJ1bl9udW1iZXIiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkiLCJsdW1pIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpX3dpdGhfbm9fb2xkZXJfdGhhbiIsImdldF9wbG90X3VybCIsInBsb3RfbmFtZSIsImdldF9jdXN0b21pemVfcGFyYW1zIiwiY3VzdG9taXplUHJvcHMiLCJzdGF0cyIsImVycm9yQmFycyIsIndpZHRoIiwiaGVpZ2h0IiwiZ2V0X3Bsb3Rfd2l0aF9vdmVybGF5Iiwib3ZlcmxheSIsImVuY29kZVVSSUNvbXBvbmVudCIsImpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJscyIsIm5vcm1hbGl6ZSIsImdldF9vdmVybGFpZWRfcGxvdHNfdXJscyIsIm92ZXJsYXlfcGxvdHMiLCJvdmVybGF5X3Bsb3QiLCJsZW5ndGgiLCJtYXAiLCJkYXRhc2V0X25hbWVfb3ZlcmxheSIsImxhYmVsIiwiZ2V0X2pyb290X3Bsb3QiLCJnZXRMdW1pc2VjdGlvbnMiLCJnZXRfdGhlX2xhdGVzdF9ydW5zIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFNQTtBQUVBLElBQU1BLE1BQVcsR0FBRztBQUNsQkMsYUFBVyxFQUFFO0FBQ1hDLFlBQVEsRUFBRSx1QkFEQztBQUVYQyxTQUFLLEVBQUU7QUFGSSxHQURLO0FBS2xCQyxZQUFVLEVBQUU7QUFDVkYsWUFBUSxFQUFDRyxxRUFBVyxFQURWO0FBRVY7QUFDQTtBQUNBO0FBQ0FGLFNBQUssRUFBRTtBQUxHO0FBTE0sQ0FBcEI7QUFjQSxJQUFNRyxnQkFBZ0IsR0FBR0MsTUFBQSxLQUE2QixNQUF0RDtBQUNBLElBQU1DLG1CQUFtQixHQUFHRCxNQUFBLEtBQXdCLE1BQXBEO0FBQ0EsSUFBTUUsd0JBQXdCLEdBQUdGLE1BQUEsS0FBNEIsTUFBN0Q7QUFDQSxJQUFNRyxrQkFBa0IsR0FBR0gsTUFBQSxLQUFzQixNQUFqRDtBQUVPLElBQU1JLGdCQUFxQixHQUFHO0FBQ25DQyxjQUFZLEVBQUU7QUFDWkEsZ0JBQVksRUFBRU4sZ0JBQWdCLElBQUksS0FEdEI7QUFFWk8sbUJBQWUsRUFBR0gsa0JBQWtCLElBQUlKLGdCQUF2QixJQUE0QyxLQUZqRDtBQUdaUSxXQUFPLEVBQUdOLG1CQUFtQixJQUFJRixnQkFBeEIsSUFBNkMsS0FIMUM7QUFJWlMsZUFBVyxFQUFHTix3QkFBd0IsSUFBSUgsZ0JBQTdCLElBQWtEO0FBSm5ELEdBRHFCO0FBT25DVSxNQUFJLEVBQUVULE9BQU8sQ0FBQ1UsR0FBUixDQUFZQyxJQUFaLElBQW9C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Q25CLFlBRGlCLEdBRXpCSCxNQUFNLENBQUNHLFlBRlg7QUFHQSxRQUFNNkIsS0FBSyxHQUFHVixPQUFPLENBQUNVLEtBQVIsR0FBZ0JWLE9BQU8sQ0FBQ1UsS0FBeEIsR0FBZ0NWLE9BQU8sQ0FBQ2QsVUFBdEQ7QUFDQSxrQ0FBdUJOLHNFQUF1QixDQUM1Q29CLE9BRDRDLENBQTlDLFNBRUlTLG9CQUZKLFNBRTJCL0IsTUFBTSxDQUFDSSxZQUZsQyxjQUVrRG1CLGtCQUFrQixDQUNsRXZCLE1BQU0sQ0FBQ2MsU0FEMkQsQ0FGcEUsdUJBSWNrQixLQUpkO0FBS0QsR0FWTSxDQUFQO0FBV0QsQ0FqQk07QUFtQkEsSUFBTUMsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixDQUFDakMsTUFBRDtBQUFBLHVDQUNMRSxzRUFBdUIsQ0FBQ0YsTUFBRCxDQURsQixTQUM2QkEsTUFBTSxDQUFDRyxZQURwQyxTQUV6QkgsTUFBTSxDQUFDSSxZQUZrQixjQUVGbUIsa0JBQWtCLENBQzFDdkIsTUFBTSxDQUFDYyxTQURtQyxDQUZoQix1Q0FJRWQsTUFBTSxDQUFDTSxZQUpUO0FBQUEsQ0FBdkI7QUFNQSxJQUFNNEIsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixDQUFDbEMsTUFBRDtBQUFBLHNDQUNQQSxNQUFNLENBQUNRLFVBREEsc0JBQ3NCUixNQUFNLENBQUNHLFlBRDdCLG1CQUVwQkgsTUFBTSxDQUFDVyxJQUZhLFNBRU5yQixnQkFBZ0IsQ0FBQ0ssSUFBakIsS0FBMEIsUUFBMUIsSUFBc0NLLE1BQU0sQ0FBQ00sWUFBN0MsMkJBQ0ZOLE1BQU0sQ0FBQ00sWUFETCxJQUVuQixFQUp5QjtBQUFBLENBQXhCO0FBT0EsSUFBTTZCLG1CQUFtQixHQUFHLFNBQXRCQSxtQkFBc0IsQ0FBQzdCLFlBQUQsRUFBMEI7QUFDM0QsbURBQTBDQSxZQUExQztBQUNELENBRk0iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguYzc1NzI0ODI1ODllMmUxZDM2N2UuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IGdldFBhdGhOYW1lIH0gZnJvbSAnLi4vY29tcG9uZW50cy91dGlscyc7XHJcbmltcG9ydCB7XHJcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXHJcbiAgVHJpcGxlUHJvcHMsXHJcbiAgTHVtaXNlY3Rpb25SZXF1ZXN0UHJvcHMsXHJcbn0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBnZXRfY3VzdG9taXplX3BhcmFtcywgZ2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMgfSBmcm9tICcuL3V0aWxzJztcclxuXHJcbmNvbnN0IGNvbmZpZzogYW55ID0ge1xyXG4gIGRldmVsb3BtZW50OiB7XHJcbiAgICByb290X3VybDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4NicsXHJcbiAgICB0aXRsZTogJ0RldmVsb3BtZW50JyxcclxuICB9LFxyXG4gIHByb2R1Y3Rpb246IHtcclxuICAgIHJvb3RfdXJsOmdldFBhdGhOYW1lKCksXHJcbiAgICAvLyByb290X3VybDogJ2h0dHBzOi8vZHFtLWd1aS53ZWIuY2Vybi5jaC9hcGkvZHFtL29mZmxpbmUvJyxcclxuICAgIC8vIHJvb3RfdXJsOiAnLi8nLFxyXG4gICAgLy8gcm9vdF91cmw6ICdodHRwOi8vbG9jYWxob3N0OjgwODYvJyxcclxuICAgIHRpdGxlOiAnT2ZmbGluZScsXHJcbiAgfSxcclxufTtcclxuXHJcbmNvbnN0IG5ld19lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5ORVdfQkFDS19FTkQgPT09ICd0cnVlJztcclxuY29uc3QgbGF5b3V0X2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52LkxBWU9VVFMgPT09ICd0cnVlJztcclxuY29uc3QgbGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTEFURVNUX1JVTlMgPT09ICd0cnVlJztcclxuY29uc3QgbHVtaXNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTFVNSVMgPT09ICd0cnVlJztcclxuXHJcbmV4cG9ydCBjb25zdCBmdW5jdGlvbnNfY29uZmlnOiBhbnkgPSB7XHJcbiAgbmV3X2JhY2tfZW5kOiB7XHJcbiAgICBuZXdfYmFja19lbmQ6IG5ld19lbnZfdmFyaWFibGUgfHwgZmFsc2UsXHJcbiAgICBsdW1pc2VjdGlvbnNfb246IChsdW1pc19lbnZfdmFyaWFibGUgJiYgbmV3X2Vudl92YXJpYWJsZSkgfHwgZmFsc2UsXHJcbiAgICBsYXlvdXRzOiAobGF5b3V0X2Vudl92YXJpYWJsZSAmJiBuZXdfZW52X3ZhcmlhYmxlKSB8fCBmYWxzZSxcclxuICAgIGxhdGVzdF9ydW5zOiAobGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlICYmIG5ld19lbnZfdmFyaWFibGUpIHx8IGZhbHNlLFxyXG4gIH0sXHJcbiAgbW9kZTogcHJvY2Vzcy5lbnYuTU9ERSB8fCAnT0ZGTElORScsXHJcbn07XHJcblxyXG5leHBvcnQgY29uc3Qgcm9vdF91cmwgPSBjb25maWdbcHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50J10ucm9vdF91cmw7XHJcblxyXG5leHBvcnQgY29uc3Qgc2VydmljZV90aXRsZSA9XHJcbiAgY29uZmlnW3Byb2Nlc3MuZW52Lk5PREVfRU5WIHx8ICdkZXZlbG9wbWVudCddLnRpdGxlO1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICBpZiAocGFyYW1zLnBsb3Rfc2VhcmNoKSB7XHJcbiAgICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZSA9IChcclxuICBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzXHJcbikgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/c2VhcmNoPSR7cGFyYW1zLnBsb3Rfc2VhcmNofSZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICAgIH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9ub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWA7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgZGF0YS9qc29uL2FyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/c2VhcmNoPSR7cGFyYW1zLnBsb3Rfc2VhcmNofWA7XHJcbiAgfVxyXG4gIHJldHVybiBgZGF0YS9qc29uL2FyY2hpdmUvJHtwYXJhbXMucnVuX251bWJlcn0ke3BhcmFtcy5kYXRhc2V0X25hbWV9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfb2xkX2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgcmV0dXJuIGBkYXRhL2pzb24vc2FtcGxlcz9tYXRjaD0ke3BhcmFtcy5kYXRhc2V0X25hbWV9JnJ1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgYXBpL3YxL3NhbXBsZXM/cnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9Jmx1bWk9JHtwYXJhbXMubHVtaX0mZGF0YXNldD0ke3BhcmFtcy5kYXRhc2V0X25hbWV9YDtcclxufTtcclxuZXhwb3J0IGNvbnN0IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4gPSAoXHJcbiAgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wc1xyXG4pID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZsdW1pPSR7cGFyYW1zLmx1bWl9JmRhdGFzZXQ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfSZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF91cmwgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgcGxvdGZhaXJ5L2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgfSR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHtwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ30/JHtnZXRfY3VzdG9taXplX3BhcmFtcyhcclxuICAgICAgcGFyYW1zLmN1c3RvbWl6ZVByb3BzXHJcbiAgICApfSR7cGFyYW1zLnN0YXRzID8gJycgOiAnc2hvd3N0YXRzPTA7J30ke3BhcmFtcy5lcnJvckJhcnMgPyAnc2hvd2VycmJhcnM9MTsnIDogJydcclxuICAgIH07dz0ke3BhcmFtcy53aWR0aH07aD0ke3BhcmFtcy5oZWlnaHR9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF93aXRoX292ZXJsYXkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgcGxvdGZhaXJ5L292ZXJsYXk/JHtnZXRfY3VzdG9taXplX3BhcmFtcyhwYXJhbXMuY3VzdG9taXplUHJvcHMpfXJlZj0ke3BhcmFtcy5vdmVybGF5XHJcbiAgICB9O29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWV9JHtwYXJhbXMuZm9sZGVyc19wYXRoXHJcbiAgICB9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KHBhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nKX0ke3BhcmFtcy5qb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHNcclxuICAgIH07JHtwYXJhbXMuc3RhdHMgPyAnJyA6ICdzaG93c3RhdHM9MDsnfSR7cGFyYW1zLmVycm9yQmFycyA/ICdzaG93ZXJyYmFycz0xOycgOiAnJ1xyXG4gICAgfW5vcm09JHtwYXJhbXMubm9ybWFsaXplfTt3PSR7cGFyYW1zLndpZHRofTtoPSR7cGFyYW1zLmhlaWdodH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgY29uc3Qgb3ZlcmxheV9wbG90cyA9XHJcbiAgICBwYXJhbXM/Lm92ZXJsYXlfcGxvdCAmJiBwYXJhbXM/Lm92ZXJsYXlfcGxvdC5sZW5ndGggPiAwXHJcbiAgICAgID8gcGFyYW1zLm92ZXJsYXlfcGxvdFxyXG4gICAgICA6IFtdO1xyXG5cclxuICByZXR1cm4gb3ZlcmxheV9wbG90cy5tYXAoKG92ZXJsYXk6IFRyaXBsZVByb3BzKSA9PiB7XHJcbiAgICBjb25zdCBkYXRhc2V0X25hbWVfb3ZlcmxheSA9IG92ZXJsYXkuZGF0YXNldF9uYW1lXHJcbiAgICAgID8gb3ZlcmxheS5kYXRhc2V0X25hbWVcclxuICAgICAgOiBwYXJhbXMuZGF0YXNldF9uYW1lO1xyXG4gICAgY29uc3QgbGFiZWwgPSBvdmVybGF5LmxhYmVsID8gb3ZlcmxheS5sYWJlbCA6IG92ZXJsYXkucnVuX251bWJlcjtcclxuICAgIHJldHVybiBgO29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMoXHJcbiAgICAgIG92ZXJsYXlcclxuICAgICl9JHtkYXRhc2V0X25hbWVfb3ZlcmxheX0ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KFxyXG4gICAgICBwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ1xyXG4gICAgKX07cmVmbGFiZWw9JHtsYWJlbH1gO1xyXG4gIH0pO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9qcm9vdF9wbG90ID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+XHJcbiAgYGpzcm9vdGZhaXJ5L2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KFxyXG4gICAgcGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmdcclxuICApfT9qc3Jvb3Q9dHJ1ZTtub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWA7XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0THVtaXNlY3Rpb25zID0gKHBhcmFtczogTHVtaXNlY3Rpb25SZXF1ZXN0UHJvcHMpID0+XHJcbiAgYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZkYXRhc2V0PSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0mbHVtaT0ke3BhcmFtcy5sdW1pfSR7ZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJyAmJiBwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICA/IGAmbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gXHJcbiAgICA6ICcnXHJcbiAgfWA7XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X3RoZV9sYXRlc3RfcnVucyA9IChub3RPbGRlclRoYW46IG51bWJlcikgPT4ge1xyXG4gIHJldHVybiBgYXBpL3YxL2xhdGVzdF9ydW5zP25vdE9sZGVyVGhhbj0ke25vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
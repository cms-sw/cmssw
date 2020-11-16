webpackHotUpdate_N_E("pages/index",{

/***/ "./config/config.ts":
/*!**************************!*\
  !*** ./config/config.ts ***!
  \**************************/
/*! exports provided: functions_config, root_url, mode, service_title, get_folders_and_plots_new_api, get_folders_and_plots_new_api_with_live_mode, get_folders_and_plots_old_api, get_run_list_by_search_old_api, get_run_list_by_search_new_api, get_run_list_by_search_new_api_with_no_older_than, get_plot_url, get_plot_with_overlay, get_overlaied_plots_urls, get_jroot_plot, getLumisections, get_the_latest_runs */
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
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_jroot_plot", function() { return get_jroot_plot; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getLumisections", function() { return getLumisections; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_the_latest_runs", function() { return get_the_latest_runs; });
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../components/utils */ "./components/utils.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils */ "./config/utils.ts");


var config = {
  development: {
    root_url: 'http://localhost:8086/',
    title: 'Development'
  },
  production: {
    root_url: "".concat(Object(_components_utils__WEBPACK_IMPORTED_MODULE_0__["getPathName"])()),
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
  mode: "ONLINE" || false
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29uZmlnL2NvbmZpZy50cyJdLCJuYW1lcyI6WyJjb25maWciLCJkZXZlbG9wbWVudCIsInJvb3RfdXJsIiwidGl0bGUiLCJwcm9kdWN0aW9uIiwiZ2V0UGF0aE5hbWUiLCJuZXdfZW52X3ZhcmlhYmxlIiwicHJvY2VzcyIsImxheW91dF9lbnZfdmFyaWFibGUiLCJsYXRlc3RfcnVuc19lbnZfdmFyaWFibGUiLCJsdW1pc19lbnZfdmFyaWFibGUiLCJlbnYiLCJMVU1JUyIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJsdW1pc2VjdGlvbnNfb24iLCJsYXlvdXRzIiwibGF0ZXN0X3J1bnMiLCJtb2RlIiwic2VydmljZV90aXRsZSIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpIiwicGFyYW1zIiwicGxvdF9zZWFyY2giLCJnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyIsImRhdGFzZXRfbmFtZSIsImZvbGRlcnNfcGF0aCIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpX3dpdGhfbGl2ZV9tb2RlIiwibm90T2xkZXJUaGFuIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkiLCJydW5fbnVtYmVyIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpIiwibHVtaSIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4iLCJnZXRfcGxvdF91cmwiLCJwbG90X25hbWUiLCJnZXRfY3VzdG9taXplX3BhcmFtcyIsImN1c3RvbWl6ZVByb3BzIiwic3RhdHMiLCJlcnJvckJhcnMiLCJ3aWR0aCIsImhlaWdodCIsImdldF9wbG90X3dpdGhfb3ZlcmxheSIsIm92ZXJsYXkiLCJlbmNvZGVVUklDb21wb25lbnQiLCJqb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMiLCJub3JtYWxpemUiLCJnZXRfb3ZlcmxhaWVkX3Bsb3RzX3VybHMiLCJvdmVybGF5X3Bsb3RzIiwib3ZlcmxheV9wbG90IiwibGVuZ3RoIiwibWFwIiwiZGF0YXNldF9uYW1lX292ZXJsYXkiLCJsYWJlbCIsImdldF9qcm9vdF9wbG90IiwiZ2V0THVtaXNlY3Rpb25zIiwiZ2V0X3RoZV9sYXRlc3RfcnVucyJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFNQTtBQUVBLElBQU1BLE1BQVcsR0FBRztBQUNsQkMsYUFBVyxFQUFFO0FBQ1hDLFlBQVEsRUFBRSx3QkFEQztBQUVYQyxTQUFLLEVBQUU7QUFGSSxHQURLO0FBS2xCQyxZQUFVLEVBQUU7QUFDVkYsWUFBUSxZQUFLRyxxRUFBVyxFQUFoQixDQURFO0FBRVZGLFNBQUssRUFBRTtBQUZHO0FBTE0sQ0FBcEI7QUFXQSxJQUFNRyxnQkFBZ0IsR0FBR0MsTUFBQSxLQUE2QixNQUF0RDtBQUNBLElBQU1DLG1CQUFtQixHQUFHRCxNQUFBLEtBQXdCLE1BQXBEO0FBQ0EsSUFBTUUsd0JBQXdCLEdBQUdGLE1BQUEsS0FBNEIsTUFBN0Q7QUFDQSxJQUFNRyxrQkFBa0IsR0FBR0gsT0FBTyxDQUFDSSxHQUFSLENBQVlDLEtBQVosS0FBc0IsTUFBakQ7QUFFTyxJQUFNQyxnQkFBcUIsR0FBRztBQUNuQ0MsY0FBWSxFQUFFO0FBQ1pBLGdCQUFZLEVBQUVSLGdCQUFnQixJQUFJLEtBRHRCO0FBRVpTLG1CQUFlLEVBQUdMLGtCQUFrQixJQUFJSixnQkFBdkIsSUFBNEMsS0FGakQ7QUFHWlUsV0FBTyxFQUFHUixtQkFBbUIsSUFBSUYsZ0JBQXhCLElBQTZDLEtBSDFDO0FBSVpXLGVBQVcsRUFBR1Isd0JBQXdCLElBQUlILGdCQUE3QixJQUFrRDtBQUpuRCxHQURxQjtBQU9uQ1ksTUFBSSxFQUFFWCxRQUFBLElBQW9CLEtB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ZWY2NzhhNjc0NWE2Mi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgZ2V0UGF0aE5hbWUgfSBmcm9tICcuLi9jb21wb25lbnRzL3V0aWxzJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBUcmlwbGVQcm9wcyxcclxuICBMdW1pc2VjdGlvblJlcXVlc3RQcm9wcyxcclxufSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IGdldF9jdXN0b21pemVfcGFyYW1zLCBnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyB9IGZyb20gJy4vdXRpbHMnO1xyXG5cclxuY29uc3QgY29uZmlnOiBhbnkgPSB7XHJcbiAgZGV2ZWxvcG1lbnQ6IHtcclxuICAgIHJvb3RfdXJsOiAnaHR0cDovL2xvY2FsaG9zdDo4MDg2LycsXHJcbiAgICB0aXRsZTogJ0RldmVsb3BtZW50JyxcclxuICB9LFxyXG4gIHByb2R1Y3Rpb246IHtcclxuICAgIHJvb3RfdXJsOiBgJHtnZXRQYXRoTmFtZSgpfWAsXHJcbiAgICB0aXRsZTogJ09ubGluZS1wbGF5YmFjaycsXHJcbiAgfSxcclxufTtcclxuXHJcbmNvbnN0IG5ld19lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5ORVdfQkFDS19FTkQgPT09ICd0cnVlJztcclxuY29uc3QgbGF5b3V0X2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52LkxBWU9VVFMgPT09ICd0cnVlJztcclxuY29uc3QgbGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTEFURVNUX1JVTlMgPT09ICd0cnVlJztcclxuY29uc3QgbHVtaXNfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTFVNSVMgPT09ICd0cnVlJztcclxuXHJcbmV4cG9ydCBjb25zdCBmdW5jdGlvbnNfY29uZmlnOiBhbnkgPSB7XHJcbiAgbmV3X2JhY2tfZW5kOiB7XHJcbiAgICBuZXdfYmFja19lbmQ6IG5ld19lbnZfdmFyaWFibGUgfHwgZmFsc2UsXHJcbiAgICBsdW1pc2VjdGlvbnNfb246IChsdW1pc19lbnZfdmFyaWFibGUgJiYgbmV3X2Vudl92YXJpYWJsZSkgfHwgZmFsc2UsXHJcbiAgICBsYXlvdXRzOiAobGF5b3V0X2Vudl92YXJpYWJsZSAmJiBuZXdfZW52X3ZhcmlhYmxlKSB8fCBmYWxzZSxcclxuICAgIGxhdGVzdF9ydW5zOiAobGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlICYmIG5ld19lbnZfdmFyaWFibGUpIHx8IGZhbHNlLFxyXG4gIH0sXHJcbiAgbW9kZTogcHJvY2Vzcy5lbnYuTU9ERSB8fCAnT0ZGTElORScsXHJcbn07XHJcblxyXG5leHBvcnQgY29uc3Qgcm9vdF91cmwgPSBjb25maWdbcHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50J10ucm9vdF91cmw7XHJcbmV4cG9ydCBjb25zdCBtb2RlID0gY29uZmlnW3Byb2Nlc3MuZW52Lk5PREVfRU5WIHx8ICdkZXZlbG9wbWVudCddLnRpdGxlO1xyXG5cclxuZXhwb3J0IGNvbnN0IHNlcnZpY2VfdGl0bGUgPVxyXG4gIGNvbmZpZ1twcm9jZXNzLmVudi5OT0RFX0VOViB8fCAnZGV2ZWxvcG1lbnQnXS50aXRsZTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgaWYgKHBhcmFtcy5wbG90X3NlYXJjaCkge1xyXG4gICAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtcclxuICAgICAgcGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGFwaS92MS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke1xyXG4gICAgcGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZSA9IChcclxuICBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzXHJcbikgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7XHJcbiAgICAgIHBhcmFtcy5kYXRhc2V0X25hbWVcclxuICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9zZWFyY2g9JHtwYXJhbXMucGxvdF9zZWFyY2h9Jm5vdE9sZGVyVGhhbj0ke1xyXG4gICAgICBwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICB9YDtcclxuICB9XHJcbiAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtcclxuICAgIHBhcmFtcy5kYXRhc2V0X25hbWVcclxuICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/bm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICBpZiAocGFyYW1zLnBsb3Rfc2VhcmNoKSB7XHJcbiAgICByZXR1cm4gYGRhdGEvanNvbi9hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P3NlYXJjaD0ke3BhcmFtcy5wbG90X3NlYXJjaH1gO1xyXG4gIH1cclxuICByZXR1cm4gYGRhdGEvanNvbi9hcmNoaXZlLyR7cGFyYW1zLnJ1bl9udW1iZXJ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgZGF0YS9qc29uL3NhbXBsZXM/bWF0Y2g9JHtwYXJhbXMuZGF0YXNldF9uYW1lfSZydW49JHtwYXJhbXMucnVuX251bWJlcn1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9zYW1wbGVzP3J1bj0ke3BhcmFtcy5ydW5fbnVtYmVyfSZsdW1pPSR7cGFyYW1zLmx1bWl9JmRhdGFzZXQ9JHtwYXJhbXMuZGF0YXNldF9uYW1lfWA7XHJcbn07XHJcbmV4cG9ydCBjb25zdCBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuID0gKFxyXG4gIHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHNcclxuKSA9PiB7XHJcbiAgcmV0dXJuIGBhcGkvdjEvc2FtcGxlcz9ydW49JHtwYXJhbXMucnVuX251bWJlcn0mbHVtaT0ke3BhcmFtcy5sdW1pfSZkYXRhc2V0PSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0mbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X3Bsb3RfdXJsID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICByZXR1cm4gYHBsb3RmYWlyeS9hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke1xyXG4gICAgcGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gIH0ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7cGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmd9PyR7Z2V0X2N1c3RvbWl6ZV9wYXJhbXMoXHJcbiAgICBwYXJhbXMuY3VzdG9taXplUHJvcHNcclxuICApfSR7cGFyYW1zLnN0YXRzID8gJycgOiAnc2hvd3N0YXRzPTA7J30ke1xyXG4gICAgcGFyYW1zLmVycm9yQmFycyA/ICdzaG93ZXJyYmFycz0xOycgOiAnJ1xyXG4gIH07dz0ke3BhcmFtcy53aWR0aH07aD0ke3BhcmFtcy5oZWlnaHR9YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF93aXRoX292ZXJsYXkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIHJldHVybiBgcGxvdGZhaXJ5L292ZXJsYXk/JHtnZXRfY3VzdG9taXplX3BhcmFtcyhwYXJhbXMuY3VzdG9taXplUHJvcHMpfXJlZj0ke1xyXG4gICAgcGFyYW1zLm92ZXJsYXlcclxuICB9O29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMocGFyYW1zKX0ke3BhcmFtcy5kYXRhc2V0X25hbWV9JHtcclxuICAgIHBhcmFtcy5mb2xkZXJzX3BhdGhcclxuICB9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KHBhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nKX0ke1xyXG4gICAgcGFyYW1zLmpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJsc1xyXG4gIH07JHtwYXJhbXMuc3RhdHMgPyAnJyA6ICdzaG93c3RhdHM9MDsnfSR7XHJcbiAgICBwYXJhbXMuZXJyb3JCYXJzID8gJ3Nob3dlcnJiYXJzPTE7JyA6ICcnXHJcbiAgfW5vcm09JHtwYXJhbXMubm9ybWFsaXplfTt3PSR7cGFyYW1zLndpZHRofTtoPSR7cGFyYW1zLmhlaWdodH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgY29uc3Qgb3ZlcmxheV9wbG90cyA9XHJcbiAgICBwYXJhbXM/Lm92ZXJsYXlfcGxvdCAmJiBwYXJhbXM/Lm92ZXJsYXlfcGxvdC5sZW5ndGggPiAwXHJcbiAgICAgID8gcGFyYW1zLm92ZXJsYXlfcGxvdFxyXG4gICAgICA6IFtdO1xyXG5cclxuICByZXR1cm4gb3ZlcmxheV9wbG90cy5tYXAoKG92ZXJsYXk6IFRyaXBsZVByb3BzKSA9PiB7XHJcbiAgICBjb25zdCBkYXRhc2V0X25hbWVfb3ZlcmxheSA9IG92ZXJsYXkuZGF0YXNldF9uYW1lXHJcbiAgICAgID8gb3ZlcmxheS5kYXRhc2V0X25hbWVcclxuICAgICAgOiBwYXJhbXMuZGF0YXNldF9uYW1lO1xyXG4gICAgY29uc3QgbGFiZWwgPSBvdmVybGF5LmxhYmVsID8gb3ZlcmxheS5sYWJlbCA6IG92ZXJsYXkucnVuX251bWJlcjtcclxuICAgIHJldHVybiBgO29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMoXHJcbiAgICAgIG92ZXJsYXlcclxuICAgICl9JHtkYXRhc2V0X25hbWVfb3ZlcmxheX0ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KFxyXG4gICAgICBwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ1xyXG4gICAgKX07cmVmbGFiZWw9JHtsYWJlbH1gO1xyXG4gIH0pO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9qcm9vdF9wbG90ID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+XHJcbiAgYGpzcm9vdGZhaXJ5L2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7XHJcbiAgICBwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgfSR7cGFyYW1zLmZvbGRlcnNfcGF0aH0vJHtlbmNvZGVVUklDb21wb25lbnQoXHJcbiAgICBwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ1xyXG4gICl9P2pzcm9vdD10cnVlO25vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW59YDtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRMdW1pc2VjdGlvbnMgPSAocGFyYW1zOiBMdW1pc2VjdGlvblJlcXVlc3RQcm9wcykgPT5cclxuICBgYXBpL3YxL3NhbXBsZXM/cnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9JmRhdGFzZXQ9JHtcclxuICAgIHBhcmFtcy5kYXRhc2V0X25hbWVcclxuICB9Jmx1bWk9JHtwYXJhbXMubHVtaX0ke1xyXG4gICAgZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJyAmJiBwYXJhbXMubm90T2xkZXJUaGFuXHJcbiAgICAgID8gYCZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWBcclxuICAgICAgOiAnJ1xyXG4gIH1gO1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF90aGVfbGF0ZXN0X3J1bnMgPSAobm90T2xkZXJUaGFuOiBudW1iZXIpID0+IHtcclxuICByZXR1cm4gYGFwaS92MS9sYXRlc3RfcnVucz9ub3RPbGRlclRoYW49JHtub3RPbGRlclRoYW59YDtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
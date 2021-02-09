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
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils */ "./config/utils.ts");


var config = {
  development: {
    root_url: 'http://localhost:8086/',
    title: 'Development'
  },
  production: {
    // root_url: `https://dqm-gui.web.cern.ch/api/dqm/offline/`,
    // root_url: 'http://localhost:8081/',
    // root_url: `${getPathName()}`,
    root_url: '/',
    title: 'Online-playback4'
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29uZmlnL2NvbmZpZy50cyJdLCJuYW1lcyI6WyJjb25maWciLCJkZXZlbG9wbWVudCIsInJvb3RfdXJsIiwidGl0bGUiLCJwcm9kdWN0aW9uIiwibmV3X2Vudl92YXJpYWJsZSIsInByb2Nlc3MiLCJsYXlvdXRfZW52X3ZhcmlhYmxlIiwibGF0ZXN0X3J1bnNfZW52X3ZhcmlhYmxlIiwibHVtaXNfZW52X3ZhcmlhYmxlIiwiZW52IiwiTFVNSVMiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIiwibGF5b3V0cyIsImxhdGVzdF9ydW5zIiwibW9kZSIsIk1PREUiLCJzZXJ2aWNlX3RpdGxlIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGkiLCJwYXJhbXMiLCJwbG90X3NlYXJjaCIsImdldFJ1bnNXaXRoTHVtaXNlY3Rpb25zIiwiZGF0YXNldF9uYW1lIiwiZm9sZGVyc19wYXRoIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUiLCJub3RPbGRlclRoYW4iLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfb2xkX2FwaSIsInJ1bl9udW1iZXIiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkiLCJsdW1pIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpX3dpdGhfbm9fb2xkZXJfdGhhbiIsImdldF9wbG90X3VybCIsInBsb3RfbmFtZSIsImdldF9jdXN0b21pemVfcGFyYW1zIiwiY3VzdG9taXplUHJvcHMiLCJzdGF0cyIsImVycm9yQmFycyIsIndpZHRoIiwiaGVpZ2h0IiwiZ2V0X3Bsb3Rfd2l0aF9vdmVybGF5Iiwib3ZlcmxheSIsImVuY29kZVVSSUNvbXBvbmVudCIsImpvaW5lZF9vdmVybGFpZWRfcGxvdHNfdXJscyIsIm5vcm1hbGl6ZSIsImdldF9vdmVybGFpZWRfcGxvdHNfdXJscyIsIm92ZXJsYXlfcGxvdHMiLCJvdmVybGF5X3Bsb3QiLCJsZW5ndGgiLCJtYXAiLCJkYXRhc2V0X25hbWVfb3ZlcmxheSIsImxhYmVsIiwiZ2V0X3Bsb3Rfd2l0aF9vdmVybGF5X25ld19hcGkiLCJsYWJlbHMiLCJvdmVybGFpZFNlcGFyYXRlbHkiLCJwbG90cyIsInBsb3RzX3N0cmluZ3MiLCJwbG90X2Zvcl9vdmVybGF5IiwicHVzaCIsImVuY29kZVVSSSIsImpvaW5lZF9wbG90cyIsImpvaW4iLCJqb2luZWRfbGFiZWxzIiwibm9ybSIsInJlZiIsImVycm9yIiwiY3VzdG9taXphdGlvbiIsInNpemVzIiwic2l6ZSIsImgiLCJ3IiwiZ2V0X2pyb290X3Bsb3QiLCJnZXRMdW1pc2VjdGlvbnMiLCJnZXRfdGhlX2xhdGVzdF9ydW5zIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBUUE7QUFFQSxJQUFNQSxNQUFXLEdBQUc7QUFDbEJDLGFBQVcsRUFBRTtBQUNYQyxZQUFRLEVBQUUsd0JBREM7QUFFWEMsU0FBSyxFQUFFO0FBRkksR0FESztBQUtsQkMsWUFBVSxFQUFFO0FBQ1Y7QUFDQTtBQUNBO0FBQ0FGLFlBQVEsRUFBRSxHQUpBO0FBS1ZDLFNBQUssRUFBRTtBQUxHO0FBTE0sQ0FBcEI7QUFjQSxJQUFNRSxnQkFBZ0IsR0FBR0MsTUFBQSxLQUE2QixNQUF0RDtBQUNBLElBQU1DLG1CQUFtQixHQUFHRCxNQUFBLEtBQXdCLE1BQXBEO0FBQ0EsSUFBTUUsd0JBQXdCLEdBQUdGLE1BQUEsS0FBNEIsTUFBN0Q7QUFDQSxJQUFNRyxrQkFBa0IsR0FBR0gsT0FBTyxDQUFDSSxHQUFSLENBQVlDLEtBQVosS0FBc0IsTUFBakQ7QUFFTyxJQUFNQyxnQkFBcUIsR0FBRztBQUNuQ0MsY0FBWSxFQUFF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K0dvQyxTQUFTLENBQUN4QyxNQUFNLENBQUNjLFNBQVIsQ0FBeEgsY0FBK0kyQixZQUEvSSxnQkFBaUt0QixLQUFqSyxnQkFBNEtDLE1BQTVLLG1CQUEyTHdCLElBQTNMLGNBQW1NM0IsS0FBbk0sU0FBMk0wQixhQUEzTSxTQUEyTkcsS0FBM04sY0FBb09DLGFBQXBPLGlCQUF3UEYsR0FBeFA7QUFDRCxHQWxCRCxNQW1CSztBQUNIO0FBQ0Q7QUFDRixDQXpCTTtBQTJCQSxJQUFNTyxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLENBQUNwRCxNQUFEO0FBQUEsdUNBQ0xFLHNFQUF1QixDQUFDRixNQUFELENBRGxCLFNBQzZCQSxNQUFNLENBQUNHLFlBRHBDLGNBRXhCSCxNQUFNLENBQUNJLFlBRmlCLGNBRURtQixrQkFBa0IsQ0FDM0N2QixNQUFNLENBQUNjLFNBRG9DLENBRmpCLDBCQUlYZCxNQUFNLENBQUNNLFlBQVAsMEJBQXNDTixNQUFNLENBQUNNLFlBQTdDLElBQThELEVBSm5EO0FBQUEsQ0FBdkI7QUFNQSxJQUFNK0MsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixDQUFDckQsTUFBRDtBQUFBLHNDQUNQQSxNQUFNLENBQUNRLFVBREEsc0JBQ3NCUixNQUFNLENBQUNHLFlBRDdCLG1CQUVwQkgsTUFBTSxDQUFDVyxJQUZhLFNBRU5wQixnQkFBZ0IsQ0FBQ0ssSUFBakIsS0FBMEIsUUFBMUIsSUFBc0NJLE1BQU0sQ0FBQ00sWUFBN0MsMkJBQ0ZOLE1BQU0sQ0FBQ00sWUFETCxJQUVuQixFQUp5QjtBQUFBLENBQXhCO0FBT0EsSUFBTWdELG1CQUFtQixHQUFHLFNBQXRCQSxtQkFBc0IsQ0FBQ2hELFlBQUQsRUFBMEI7QUFDM0QsbURBQTBDQSxZQUExQztBQUNELENBRk0iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMzNlNWNiYjIzYTdhMmQ0NDhhNmUuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IHNpemVzIH0gZnJvbSAnLi4vY29tcG9uZW50cy9jb25zdGFudHMnO1xyXG5pbXBvcnQgeyBnZXRQYXRoTmFtZSB9IGZyb20gJy4uL2NvbXBvbmVudHMvdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFRyaXBsZVByb3BzLFxyXG4gIEx1bWlzZWN0aW9uUmVxdWVzdFByb3BzLFxyXG59IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgUGFyYW1ldGVyc0ZvckFwaSwgUGxvdFByb3BlcnRpZXMgfSBmcm9tICcuLi9wbG90c0xvY2FsT3ZlcmxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgZ2V0X2N1c3RvbWl6ZV9wYXJhbXMsIGdldFJ1bnNXaXRoTHVtaXNlY3Rpb25zIH0gZnJvbSAnLi91dGlscyc7XHJcblxyXG5jb25zdCBjb25maWc6IGFueSA9IHtcclxuICBkZXZlbG9wbWVudDoge1xyXG4gICAgcm9vdF91cmw6ICdodHRwOi8vbG9jYWxob3N0OjgwODYvJyxcclxuICAgIHRpdGxlOiAnRGV2ZWxvcG1lbnQnLFxyXG4gIH0sXHJcbiAgcHJvZHVjdGlvbjoge1xyXG4gICAgLy8gcm9vdF91cmw6IGBodHRwczovL2RxbS1ndWkud2ViLmNlcm4uY2gvYXBpL2RxbS9vZmZsaW5lL2AsXHJcbiAgICAvLyByb290X3VybDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4MS8nLFxyXG4gICAgLy8gcm9vdF91cmw6IGAke2dldFBhdGhOYW1lKCl9YCxcclxuICAgIHJvb3RfdXJsOiAnLycsXHJcbiAgICB0aXRsZTogJ09ubGluZS1wbGF5YmFjazQnLFxyXG4gIH0sXHJcbn07XHJcblxyXG5jb25zdCBuZXdfZW52X3ZhcmlhYmxlID0gcHJvY2Vzcy5lbnYuTkVXX0JBQ0tfRU5EID09PSAndHJ1ZSc7XHJcbmNvbnN0IGxheW91dF9lbnZfdmFyaWFibGUgPSBwcm9jZXNzLmVudi5MQVlPVVRTID09PSAndHJ1ZSc7XHJcbmNvbnN0IGxhdGVzdF9ydW5zX2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52LkxBVEVTVF9SVU5TID09PSAndHJ1ZSc7XHJcbmNvbnN0IGx1bWlzX2Vudl92YXJpYWJsZSA9IHByb2Nlc3MuZW52LkxVTUlTID09PSAndHJ1ZSc7XHJcblxyXG5leHBvcnQgY29uc3QgZnVuY3Rpb25zX2NvbmZpZzogYW55ID0ge1xyXG4gIG5ld19iYWNrX2VuZDoge1xyXG4gICAgbmV3X2JhY2tfZW5kOiBuZXdfZW52X3ZhcmlhYmxlIHx8IGZhbHNlLFxyXG4gICAgbHVtaXNlY3Rpb25zX29uOiAobHVtaXNfZW52X3ZhcmlhYmxlICYmIG5ld19lbnZfdmFyaWFibGUpIHx8IGZhbHNlLFxyXG4gICAgbGF5b3V0czogKGxheW91dF9lbnZfdmFyaWFibGUgJiYgbmV3X2Vudl92YXJpYWJsZSkgfHwgZmFsc2UsXHJcbiAgICBsYXRlc3RfcnVuczogKGxhdGVzdF9ydW5zX2Vudl92YXJpYWJsZSAmJiBuZXdfZW52X3ZhcmlhYmxlKSB8fCBmYWxzZSxcclxuICB9LFxyXG4gIG1vZGU6IHByb2Nlc3MuZW52Lk1PREUgfHwgJ09GRkxJTkUnLFxyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IHJvb3RfdXJsID0gY29uZmlnW3Byb2Nlc3MuZW52Lk5PREVfRU5WIHx8ICdkZXZlbG9wbWVudCddLnJvb3RfdXJsO1xyXG5leHBvcnQgY29uc3QgbW9kZSA9IGNvbmZpZ1twcm9jZXNzLmVudi5OT0RFX0VOViB8fCAnZGV2ZWxvcG1lbnQnXS50aXRsZTtcclxuXHJcbmV4cG9ydCBjb25zdCBzZXJ2aWNlX3RpdGxlID1cclxuICBjb25maWdbcHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50J10udGl0bGU7XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xyXG4gIGlmIChwYXJhbXMucGxvdF9zZWFyY2gpIHtcclxuICAgIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgICB9LyR7cGFyYW1zLmZvbGRlcnNfcGF0aH0/c2VhcmNoPSR7cGFyYW1zLnBsb3Rfc2VhcmNofWA7XHJcbiAgfVxyXG4gIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9YDtcclxufTtcclxuZXhwb3J0IGNvbnN0IGdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpX3dpdGhfbGl2ZV9tb2RlID0gKFxyXG4gIHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHNcclxuKSA9PiB7XHJcbiAgaWYgKHBhcmFtcy5wbG90X3NlYXJjaCkge1xyXG4gICAgcmV0dXJuIGBhcGkvdjEvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgICAgIH0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9zZWFyY2g9JHtwYXJhbXMucGxvdF9zZWFyY2h9Jm5vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW5cclxuICAgICAgfWA7XHJcbiAgfVxyXG4gIHJldHVybiBgYXBpL3YxL2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9P25vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW59YDtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfZm9sZGVyc19hbmRfcGxvdHNfb2xkX2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgaWYgKHBhcmFtcy5wbG90X3NlYXJjaCkge1xyXG4gICAgcmV0dXJuIGBkYXRhL2pzb24vYXJjaGl2ZS8ke3BhcmFtcy5ydW5fbnVtYmVyfSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0vJHtwYXJhbXMuZm9sZGVyc19wYXRofT9zZWFyY2g9JHtwYXJhbXMucGxvdF9zZWFyY2h9YDtcclxuICB9XHJcbiAgcmV0dXJuIGBkYXRhL2pzb24vYXJjaGl2ZS8ke3BhcmFtcy5ydW5fbnVtYmVyfSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0vJHtwYXJhbXMuZm9sZGVyc19wYXRofWA7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcclxuICByZXR1cm4gYGRhdGEvanNvbi9zYW1wbGVzP21hdGNoPSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0mcnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9YDtcclxufTtcclxuZXhwb3J0IGNvbnN0IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgcmV0dXJuIGBhcGkvdjEvc2FtcGxlcz9ydW49JHtwYXJhbXMucnVuX251bWJlcn0mbHVtaT0ke3BhcmFtcy5sdW1pfSZkYXRhc2V0PSR7cGFyYW1zLmRhdGFzZXRfbmFtZX1gO1xyXG59O1xyXG5leHBvcnQgY29uc3QgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpX3dpdGhfbm9fb2xkZXJfdGhhbiA9IChcclxuICBwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzXHJcbikgPT4ge1xyXG4gIHJldHVybiBgYXBpL3YxL3NhbXBsZXM/cnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9Jmx1bWk9JHtwYXJhbXMubHVtaX0mZGF0YXNldD0ke3BhcmFtcy5kYXRhc2V0X25hbWV9Jm5vdE9sZGVyVGhhbj0ke3BhcmFtcy5ub3RPbGRlclRoYW59YDtcclxufTtcclxuZXhwb3J0IGNvbnN0IGdldF9wbG90X3VybCA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzICYgUGFyYW1ldGVyc0ZvckFwaSAmIGFueSkgPT4ge1xyXG4gIHJldHVybiBgcGxvdGZhaXJ5L2FyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZVxyXG4gICAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7cGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmd9PyR7Z2V0X2N1c3RvbWl6ZV9wYXJhbXMoXHJcbiAgICAgIHBhcmFtcy5jdXN0b21pemVQcm9wc1xyXG4gICAgKX0ke3BhcmFtcy5zdGF0cyA/ICcnIDogJ3Nob3dzdGF0cz0wJ30ke3BhcmFtcy5lcnJvckJhcnMgPyAnc2hvd2VycmJhcnM9MScgOiAnJ1xyXG4gICAgfTt3PSR7cGFyYW1zLndpZHRofTtoPSR7cGFyYW1zLmhlaWdodH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9wbG90X3dpdGhfb3ZlcmxheSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgcmV0dXJuIGBwbG90ZmFpcnkvb3ZlcmxheT8ke2dldF9jdXN0b21pemVfcGFyYW1zKHBhcmFtcy5jdXN0b21pemVQcm9wcyl9cmVmPSR7cGFyYW1zLm92ZXJsYXlcclxuICAgIH07b2JqPWFyY2hpdmUvJHtnZXRSdW5zV2l0aEx1bWlzZWN0aW9ucyhwYXJhbXMpfSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0vJHtwYXJhbXMuZm9sZGVyc19wYXRoXHJcbiAgICB9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KHBhcmFtcy5wbG90X25hbWUgYXMgc3RyaW5nKX0ke3BhcmFtcy5qb2luZWRfb3ZlcmxhaWVkX3Bsb3RzX3VybHNcclxuICAgIH07JHtwYXJhbXMuc3RhdHMgPyAnJyA6ICdzaG93c3RhdHM9MDsnfSR7cGFyYW1zLmVycm9yQmFycyA/ICdzaG93ZXJyYmFycz0xOycgOiAnJ1xyXG4gICAgfW5vcm09JHtwYXJhbXMubm9ybWFsaXplfTt3PSR7cGFyYW1zLndpZHRofTtoPSR7cGFyYW1zLmhlaWdodH1gO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9vdmVybGFpZWRfcGxvdHNfdXJscyA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XHJcbiAgY29uc3Qgb3ZlcmxheV9wbG90cyA9XHJcbiAgICBwYXJhbXM/Lm92ZXJsYXlfcGxvdCAmJiBwYXJhbXM/Lm92ZXJsYXlfcGxvdC5sZW5ndGggPiAwXHJcbiAgICAgID8gcGFyYW1zLm92ZXJsYXlfcGxvdFxyXG4gICAgICA6IFtdO1xyXG5cclxuICByZXR1cm4gb3ZlcmxheV9wbG90cy5tYXAoKG92ZXJsYXk6IFRyaXBsZVByb3BzKSA9PiB7XHJcbiAgICBjb25zdCBkYXRhc2V0X25hbWVfb3ZlcmxheSA9IG92ZXJsYXkuZGF0YXNldF9uYW1lXHJcbiAgICAgID8gb3ZlcmxheS5kYXRhc2V0X25hbWVcclxuICAgICAgOiBwYXJhbXMuZGF0YXNldF9uYW1lO1xyXG4gICAgY29uc3QgbGFiZWwgPSBvdmVybGF5LmxhYmVsID8gb3ZlcmxheS5sYWJlbCA6IG92ZXJsYXkucnVuX251bWJlcjtcclxuICAgIHJldHVybiBgO29iaj1hcmNoaXZlLyR7Z2V0UnVuc1dpdGhMdW1pc2VjdGlvbnMoXHJcbiAgICAgIG92ZXJsYXlcclxuICAgICl9JHtkYXRhc2V0X25hbWVfb3ZlcmxheX0ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KFxyXG4gICAgICBwYXJhbXMucGxvdF9uYW1lIGFzIHN0cmluZ1xyXG4gICAgKX07cmVmbGFiZWw9JHtsYWJlbH1gO1xyXG4gIH0pO1xyXG59O1xyXG5cclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfcGxvdF93aXRoX292ZXJsYXlfbmV3X2FwaSA9IChwYXJhbXM6IFBhcmFtZXRlcnNGb3JBcGkpID0+IHtcclxuICAvL2VtcHR5IHN0cmluZyBpbiBvcmRlciB0byBzZXQgJnJlZmxhYmVsPSBpbiB0aGUgc3RhcnQgb2Ygam9pbmVkX2xhYmVscyBzdHJpbmdcclxuICBjb25zdCBsYWJlbHM6IHN0cmluZ1tdID0gWycnXVxyXG4gIGlmIChwYXJhbXMub3ZlcmxhaWRTZXBhcmF0ZWx5Py5wbG90cykge1xyXG4gICAgY29uc3QgcGxvdHNfc3RyaW5ncyA9IHBhcmFtcy5vdmVybGFpZFNlcGFyYXRlbHkucGxvdHMubWFwKChwbG90X2Zvcl9vdmVybGF5OiBQbG90UHJvcGVydGllcykgPT4ge1xyXG4gICAgICBsYWJlbHMucHVzaChwbG90X2Zvcl9vdmVybGF5LmxhYmVsID8gcGxvdF9mb3Jfb3ZlcmxheS5sYWJlbCA6IHBhcmFtcy5ydW5fbnVtYmVyKVxyXG4gICAgICByZXR1cm4gKGBvYmo9YXJjaGl2ZS8ke3BhcmFtcy5ydW5fbnVtYmVyfSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0vJHtwbG90X2Zvcl9vdmVybGF5LmZvbGRlcnNfcGF0aH0vJHsoZW5jb2RlVVJJKHBsb3RfZm9yX292ZXJsYXkucGxvdF9uYW1lKSl9YClcclxuICAgIH0pXHJcbiAgICBjb25zdCBqb2luZWRfcGxvdHMgPSBwbG90c19zdHJpbmdzLmpvaW4oJyYnKVxyXG4gICAgY29uc3Qgam9pbmVkX2xhYmVscyA9IGxhYmVscy5qb2luKCcmcmVmbGFiZWw9JylcclxuICAgIGNvbnN0IG5vcm0gPSBwYXJhbXMubm9ybWFsaXplXHJcbiAgICBjb25zdCBzdGF0cyA9IHBhcmFtcy5zdGF0cyA/ICcnIDogJ3N0YXRzPTAnXHJcbiAgICBjb25zdCByZWYgPSBwYXJhbXMub3ZlcmxhaWRTZXBhcmF0ZWx5LnJlZiA/IHBhcmFtcy5vdmVybGFpZFNlcGFyYXRlbHkucmVmIDogJ292ZXJsYXknXHJcbiAgICBjb25zdCBlcnJvciA9IHBhcmFtcy5lcnJvciA/ICcmc2hvd2VycmJhcnM9MScgOiAnJ1xyXG4gICAgY29uc3QgY3VzdG9taXphdGlvbiA9IGdldF9jdXN0b21pemVfcGFyYW1zKHBhcmFtcy5jdXN0b21pemVQcm9wcylcclxuICAgIC8vQHRzLWlnbm9yZVxyXG4gICAgY29uc3QgaGVpZ2h0ID0gc2l6ZXNbcGFyYW1zLnNpemVdLnNpemUuaFxyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBjb25zdCB3aWR0aCA9IHNpemVzW3BhcmFtcy5zaXplXS5zaXplLndcclxuXHJcbiAgICByZXR1cm4gYGFwaS92MS9yZW5kZXJfb3ZlcmxheT9vYmo9YXJjaGl2ZS8ke3BhcmFtcy5ydW5fbnVtYmVyfSR7cGFyYW1zLmRhdGFzZXRfbmFtZX0vJHtwYXJhbXMuZm9sZGVyc19wYXRofS8keyhlbmNvZGVVUkkocGFyYW1zLnBsb3RfbmFtZSkpfSYke2pvaW5lZF9wbG90c30mdz0ke3dpZHRofSZoPSR7aGVpZ2h0fSZub3JtPSR7bm9ybX0mJHtzdGF0c30ke2pvaW5lZF9sYWJlbHN9JHtlcnJvcn0mJHtjdXN0b21pemF0aW9ufXJlZj0ke3JlZn1gXHJcbiAgfVxyXG4gIGVsc2Uge1xyXG4gICAgcmV0dXJuXHJcbiAgfVxyXG59XHJcblxyXG5leHBvcnQgY29uc3QgZ2V0X2pyb290X3Bsb3QgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT5cclxuICBganNyb290ZmFpcnkvYXJjaGl2ZS8ke2dldFJ1bnNXaXRoTHVtaXNlY3Rpb25zKHBhcmFtcyl9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgfS8ke3BhcmFtcy5mb2xkZXJzX3BhdGh9LyR7ZW5jb2RlVVJJQ29tcG9uZW50KFxyXG4gICAgcGFyYW1zLnBsb3RfbmFtZSBhcyBzdHJpbmdcclxuICApfT9qc3Jvb3Q9dHJ1ZTske3BhcmFtcy5ub3RPbGRlclRoYW4gPyBgbm90T2xkZXJUaGFuPSR7cGFyYW1zLm5vdE9sZGVyVGhhbn1gIDogJyd9YDtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRMdW1pc2VjdGlvbnMgPSAocGFyYW1zOiBMdW1pc2VjdGlvblJlcXVlc3RQcm9wcykgPT5cclxuICBgYXBpL3YxL3NhbXBsZXM/cnVuPSR7cGFyYW1zLnJ1bl9udW1iZXJ9JmRhdGFzZXQ9JHtwYXJhbXMuZGF0YXNldF9uYW1lXHJcbiAgfSZsdW1pPSR7cGFyYW1zLmx1bWl9JHtmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnICYmIHBhcmFtcy5ub3RPbGRlclRoYW5cclxuICAgID8gYCZub3RPbGRlclRoYW49JHtwYXJhbXMubm90T2xkZXJUaGFufWBcclxuICAgIDogJydcclxuICB9YDtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfdGhlX2xhdGVzdF9ydW5zID0gKG5vdE9sZGVyVGhhbjogbnVtYmVyKSA9PiB7XHJcbiAgcmV0dXJuIGBhcGkvdjEvbGF0ZXN0X3J1bnM/bm90T2xkZXJUaGFuPSR7bm90T2xkZXJUaGFufWA7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
webpackHotUpdate_N_E("pages/_app",{

/***/ "./components/constants.ts":
/*!*********************************!*\
  !*** ./components/constants.ts ***!
  \*********************************/
/*! exports provided: sizes, field_name, FOLDERS_OR_PLOTS_REDUCER, NAV_REDUCER, REFERENCE_REDCER, overlayOptions, xyzTypes, withReference, dataSetSelections, viewPositions, plotsProportionsOptions, additional_run_info, main_run_info, run_info */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "sizes", function() { return sizes; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "field_name", function() { return field_name; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "FOLDERS_OR_PLOTS_REDUCER", function() { return FOLDERS_OR_PLOTS_REDUCER; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "NAV_REDUCER", function() { return NAV_REDUCER; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "REFERENCE_REDCER", function() { return REFERENCE_REDCER; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "overlayOptions", function() { return overlayOptions; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "xyzTypes", function() { return xyzTypes; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "withReference", function() { return withReference; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "dataSetSelections", function() { return dataSetSelections; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "viewPositions", function() { return viewPositions; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "plotsProportionsOptions", function() { return plotsProportionsOptions; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "additional_run_info", function() { return additional_run_info; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "main_run_info", function() { return main_run_info; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "run_info", function() { return run_info; });
var sizes = {
  tiny: {
    label: 'Tiny',
    size: {
      w: 67,
      h: 50
    }
  },
  small: {
    label: 'Small',
    size: {
      w: 133,
      h: 100
    }
  },
  medium: {
    label: 'Medium',
    size: {
      w: 266,
      h: 200
    }
  },
  large: {
    label: 'Large',
    size: {
      w: 532,
      h: 400
    }
  },
  fill: {
    label: 'Fill',
    size: {
      w: 720,
      h: 541
    }
  }
};
var field_name = {
  dataset_name: 'Dataset name',
  run_number: 'Run number',
  label: 'label'
};
var FOLDERS_OR_PLOTS_REDUCER = {
  SET_PLOT_TO_OVERLAY: 'SET_PLOT_TO_OVERLAY',
  SET_WIDTH: 'SET_WIDTH',
  SET_HEIGHT: 'SET_HEIGHT',
  SET_ZOOMED_PLOT_SIZE: 'SET_ZOOMED_PLOT_SIZE',
  SET_NORMALIZE: 'SET_NORMALIZE',
  SET_STATS: 'SET_STATS',
  SET_ERR_BARS: 'SET_ERR_BARS',
  SHOW: 'SHOW',
  JSROOT_MODE: 'JSROOT_MODE',
  SET_PARAMS_FOR_CUSTOMIZE: 'SET_PARAMS_FOR_CUSTOMIZE'
};
var NAV_REDUCER = {
  SET_SEARCH_BY_DATASET_NAME: 'SET_SEARCH_BY_DATASET_NAME',
  SET_SEARCH_BY_RUN_NUMBER: 'SET_SEARCH_BY_RUN_NUMBER'
};
var REFERENCE_REDCER = {
  CHANGE_TRIPLES_VALUES: 'CHANGE_TRIPLES_VALUES',
  OPEN_MODAL: 'OPEN_MODAL'
};
var overlayOptions = [{
  label: 'Overlay',
  value: 'overlay'
}, {
  label: 'On side',
  value: 'onSide'
}, {
  label: 'Overlay+ratio',
  value: 'ratiooverlay'
}, {
  label: 'Stacked',
  value: 'stacked'
}];
var xyzTypes = [{
  label: 'Default',
  value: ''
}, {
  label: 'Linear',
  value: 'lin'
}, {
  label: 'Log',
  value: 'log'
}];
var withReference = [{
  label: 'Default',
  value: ''
}, {
  label: 'Yes',
  value: 'yes'
}, {
  label: 'No',
  value: 'no'
}];
var dataSetSelections = [{
  label: 'Dataset Select',
  value: 'datasetSelect'
}, {
  label: 'Dataset Builder',
  value: 'datasetBuilder'
}];
var viewPositions = [{
  label: 'Horizontal',
  value: 'horizontal'
}, {
  label: 'Vertical',
  value: 'vertical'
}];
var plotsProportionsOptions = [{
  label: '50% : 50%',
  value: '50%'
}, {
  label: '25% : 75%',
  value: '25%'
}];
var additional_run_info = [{
  value: 'CMSSW_Version',
  label: 'CMSSW version: '
}, {
  value: 'CertificationSummary',
  label: 'CertificationSummary: '
}, {
  value: 'hostName',
  label: 'Host name: '
}, {
  value: 'iEvent',
  label: 'Event #: '
}, {
  value: 'processID',
  label: 'Process ID: '
}, {
  value: 'processLatency',
  label: 'Process Latency: '
}, {
  value: 'processName',
  label: 'Process Name: '
}, {
  value: 'processStartTimeStamp',
  label: 'Process Start Time, UTC time: ',
  type: 'time'
}, {
  value: 'processTimeStamp',
  label: 'Process Time, UTC time: ',
  type: 'time'
}, {
  value: 'processedEvents',
  label: 'Processed Events: '
}, {
  value: 'reportSummary',
  label: 'Report Summary: '
}, {
  value: 'runStartTimeStamp',
  label: 'Run started, UTC time: ',
  type: 'time'
}, {
  value: 'workingDir',
  label: 'Working directory: '
}];
var main_run_info = [{
  value: 'iRun',
  label: 'Run: '
}, {
  value: 'iLumiSection',
  label: 'LS #: '
}, {
  value: 'iEvent',
  label: 'Event #: '
}, {
  value: 'runStartTimeStamp',
  label: 'Run started, UTC time: ',
  type: 'time'
}];
var run_info = main_run_info.concat(additional_run_info);

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9jb25zdGFudHMudHMiXSwibmFtZXMiOlsic2l6ZXMiLCJ0aW55IiwibGFiZWwiLCJzaXplIiwidyIsImgiLCJzbWFsbCIsIm1lZGl1bSIsImxhcmdlIiwiZmlsbCIsImZpZWxkX25hbWUiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwiRk9MREVSU19PUl9QTE9UU19SRURVQ0VSIiwiU0VUX1BMT1RfVE9fT1ZFUkxBWSIsIlNFVF9XSURUSCIsIlNFVF9IRUlHSFQiLCJTRVRfWk9PTUVEX1BMT1RfU0laRSIsIlNFVF9OT1JNQUxJWkUiLCJTRVRfU1RBVFMiLCJTRVRfRVJSX0JBUlMiLCJTSE9XIiwiSlNST09UX01PREUiLCJTRVRfUEFSQU1TX0ZPUl9DVVNUT01JWkUiLCJOQVZfUkVEVUNFUiIsIlNFVF9TRUFSQ0hfQllfREFUQVNFVF9OQU1FIiwiU0VUX1NFQVJDSF9CWV9SVU5fTlVNQkVSIiwiUkVGRVJFTkNFX1JFRENFUiIsIkNIQU5HRV9UUklQTEVTX1ZBTFVFUyIsIk9QRU5fTU9EQUwiLCJvdmVybGF5T3B0aW9ucyIsInZhbHVlIiwieHl6VHlwZXMiLCJ3aXRoUmVmZXJlbmNlIiwiZGF0YVNldFNlbGVjdGlvbnMiLCJ2aWV3UG9zaXRpb25zIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJhZGRpdGlvbmFsX3J1bl9pbmZvIiwidHlwZSIsIm1haW5fcnVuX2luZm8iLCJydW5faW5mbyIsImNvbmNhdCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFPLElBQU1BLEtBQUssR0FBRztBQUNuQkMsTUFBSSxFQUFFO0FBQ0pDLFNBQUssRUFBRSxNQURIO0FBRUpDLFFBQUksRUFBRTtBQUNKQyxPQUFDLEVBQUUsRUFEQztBQUVKQyxPQUFDLEVBQUU7QUFGQztBQUZG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O0FBQUU2QixPQUFLLEVBQUUsZUFBVDtBQUEwQjdCLE9BQUssRUFBRTtBQUFqQyxDQW5CaUMsRUFvQmpDO0FBQ0U2QixPQUFLLEVBQUUsbUJBRFQ7QUFFRTdCLE9BQUssRUFBRSx5QkFGVDtBQUdFb0MsTUFBSSxFQUFFO0FBSFIsQ0FwQmlDLEVBeUJqQztBQUFFUCxPQUFLLEVBQUUsWUFBVDtBQUF1QjdCLE9BQUssRUFBRTtBQUE5QixDQXpCaUMsQ0FBNUI7QUE0QkEsSUFBTXFDLGFBQWEsR0FBRyxDQUMzQjtBQUFFUixPQUFLLEVBQUUsTUFBVDtBQUFpQjdCLE9BQUssRUFBRTtBQUF4QixDQUQyQixFQUUzQjtBQUFFNkIsT0FBSyxFQUFFLGNBQVQ7QUFBeUI3QixPQUFLLEVBQUU7QUFBaEMsQ0FGMkIsRUFHM0I7QUFBRTZCLE9BQUssRUFBRSxRQUFUO0FBQW1CN0IsT0FBSyxFQUFFO0FBQTFCLENBSDJCLEVBSTNCO0FBQUU2QixPQUFLLEVBQUUsbUJBQVQ7QUFBOEI3QixPQUFLLEVBQUUseUJBQXJDO0FBQWdFb0MsTUFBSSxFQUFFO0FBQXRFLENBSjJCLENBQXRCO0FBT0EsSUFBTUUsUUFBUSxHQUFHRCxhQUFhLENBQUNFLE1BQWQsQ0FBcUJKLG1CQUFyQixDQUFqQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9fYXBwLmYxYjU2YjNiNjQ2NmYyNThmZjY4LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJleHBvcnQgY29uc3Qgc2l6ZXMgPSB7XG4gIHRpbnk6IHtcbiAgICBsYWJlbDogJ1RpbnknLFxuICAgIHNpemU6IHtcbiAgICAgIHc6IDY3LFxuICAgICAgaDogNTAsXG4gICAgfSxcbiAgfSxcbiAgc21hbGw6IHtcbiAgICBsYWJlbDogJ1NtYWxsJyxcbiAgICBzaXplOiB7XG4gICAgICB3OiAxMzMsXG4gICAgICBoOiAxMDAsXG4gICAgfSxcbiAgfSxcbiAgbWVkaXVtOiB7XG4gICAgbGFiZWw6ICdNZWRpdW0nLFxuICAgIHNpemU6IHtcbiAgICAgIHc6IDI2NixcbiAgICAgIGg6IDIwMCxcbiAgICB9LFxuICB9LFxuICBsYXJnZToge1xuICAgIGxhYmVsOiAnTGFyZ2UnLFxuICAgIHNpemU6IHtcbiAgICAgIHc6IDUzMixcbiAgICAgIGg6IDQwMCxcbiAgICB9LFxuICB9LFxuICBmaWxsOiB7XG4gICAgbGFiZWw6ICdGaWxsJyxcbiAgICBzaXplOiB7XG4gICAgICB3OiA3MjAsXG4gICAgICBoOiA1NDEsXG4gICAgfSxcbiAgfSxcbn07XG5cbmV4cG9ydCBjb25zdCBmaWVsZF9uYW1lOiBhbnkgPSB7XG4gIGRhdGFzZXRfbmFtZTogJ0RhdGFzZXQgbmFtZScsXG4gIHJ1bl9udW1iZXI6ICdSdW4gbnVtYmVyJyxcbiAgbGFiZWw6ICdsYWJlbCcsXG59O1xuXG5leHBvcnQgY29uc3QgRk9MREVSU19PUl9QTE9UU19SRURVQ0VSID0ge1xuICBTRVRfUExPVF9UT19PVkVSTEFZOiAnU0VUX1BMT1RfVE9fT1ZFUkxBWScsXG4gIFNFVF9XSURUSDogJ1NFVF9XSURUSCcsXG4gIFNFVF9IRUlHSFQ6ICdTRVRfSEVJR0hUJyxcbiAgU0VUX1pPT01FRF9QTE9UX1NJWkU6ICdTRVRfWk9PTUVEX1BMT1RfU0laRScsXG4gIFNFVF9OT1JNQUxJWkU6ICdTRVRfTk9STUFMSVpFJyxcbiAgU0VUX1NUQVRTOiAnU0VUX1NUQVRTJyxcbiAgU0VUX0VSUl9CQVJTOiAnU0VUX0VSUl9CQVJTJyxcbiAgU0hPVzogJ1NIT1cnLFxuICBKU1JPT1RfTU9ERTogJ0pTUk9PVF9NT0RFJyxcbiAgU0VUX1BBUkFNU19GT1JfQ1VTVE9NSVpFOiAnU0VUX1BBUkFNU19GT1JfQ1VTVE9NSVpFJyxcbn07XG5cbmV4cG9ydCBjb25zdCBOQVZfUkVEVUNFUiA9IHtcbiAgU0VUX1NFQVJDSF9CWV9EQVRBU0VUX05BTUU6ICdTRVRfU0VBUkNIX0JZX0RBVEFTRVRfTkFNRScsXG4gIFNFVF9TRUFSQ0hfQllfUlVOX05VTUJFUjogJ1NFVF9TRUFSQ0hfQllfUlVOX05VTUJFUicsXG59O1xuXG5leHBvcnQgY29uc3QgUkVGRVJFTkNFX1JFRENFUiA9IHtcbiAgQ0hBTkdFX1RSSVBMRVNfVkFMVUVTOiAnQ0hBTkdFX1RSSVBMRVNfVkFMVUVTJyxcbiAgT1BFTl9NT0RBTDogJ09QRU5fTU9EQUwnLFxufTtcblxuZXhwb3J0IGNvbnN0IG92ZXJsYXlPcHRpb25zID0gW1xuICB7IGxhYmVsOiAnT3ZlcmxheScsIHZhbHVlOiAnb3ZlcmxheScgfSxcbiAgeyBsYWJlbDogJ09uIHNpZGUnLCB2YWx1ZTogJ29uU2lkZScgfSxcbiAgeyBsYWJlbDogJ092ZXJsYXkrcmF0aW8nLCB2YWx1ZTogJ3JhdGlvb3ZlcmxheScgfSxcbiAgeyBsYWJlbDogJ1N0YWNrZWQnLCB2YWx1ZTogJ3N0YWNrZWQnIH0sXG5dO1xuXG5leHBvcnQgY29uc3QgeHl6VHlwZXMgPSBbXG4gIHsgbGFiZWw6ICdEZWZhdWx0JywgdmFsdWU6ICcnIH0sXG4gIHsgbGFiZWw6ICdMaW5lYXInLCB2YWx1ZTogJ2xpbicgfSxcbiAgeyBsYWJlbDogJ0xvZycsIHZhbHVlOiAnbG9nJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHdpdGhSZWZlcmVuY2UgPSBbXG4gIHsgbGFiZWw6ICdEZWZhdWx0JywgdmFsdWU6ICcnIH0sXG4gIHsgbGFiZWw6ICdZZXMnLCB2YWx1ZTogJ3llcycgfSxcbiAgeyBsYWJlbDogJ05vJywgdmFsdWU6ICdubycgfSxcbl07XG5cbmV4cG9ydCBjb25zdCBkYXRhU2V0U2VsZWN0aW9ucyA9IFtcbiAge1xuICAgIGxhYmVsOiAnRGF0YXNldCBTZWxlY3QnLFxuICAgIHZhbHVlOiAnZGF0YXNldFNlbGVjdCcsXG4gIH0sXG4gIHtcbiAgICBsYWJlbDogJ0RhdGFzZXQgQnVpbGRlcicsXG4gICAgdmFsdWU6ICdkYXRhc2V0QnVpbGRlcicsXG4gIH0sXG5dO1xuXG5leHBvcnQgY29uc3Qgdmlld1Bvc2l0aW9ucyA9IFtcbiAgeyBsYWJlbDogJ0hvcml6b250YWwnLCB2YWx1ZTogJ2hvcml6b250YWwnIH0sXG4gIHsgbGFiZWw6ICdWZXJ0aWNhbCcsIHZhbHVlOiAndmVydGljYWwnIH0sXG5dO1xuXG5leHBvcnQgY29uc3QgcGxvdHNQcm9wb3J0aW9uc09wdGlvbnMgPSBbXG4gIHsgbGFiZWw6ICc1MCUgOiA1MCUnLCB2YWx1ZTogJzUwJScgfSxcbiAgeyBsYWJlbDogJzI1JSA6IDc1JScsIHZhbHVlOiAnMjUlJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IGFkZGl0aW9uYWxfcnVuX2luZm8gPSBbXG4gIHsgdmFsdWU6ICdDTVNTV19WZXJzaW9uJywgbGFiZWw6ICdDTVNTVyB2ZXJzaW9uOiAnIH0sXG4gIHsgdmFsdWU6ICdDZXJ0aWZpY2F0aW9uU3VtbWFyeScsIGxhYmVsOiAnQ2VydGlmaWNhdGlvblN1bW1hcnk6ICcgfSxcbiAgeyB2YWx1ZTogJ2hvc3ROYW1lJywgbGFiZWw6ICdIb3N0IG5hbWU6ICcgfSxcbiAgeyB2YWx1ZTogJ2lFdmVudCcsIGxhYmVsOiAnRXZlbnQgIzogJyB9LFxuICB7IHZhbHVlOiAncHJvY2Vzc0lEJywgbGFiZWw6ICdQcm9jZXNzIElEOiAnIH0sXG4gIHsgdmFsdWU6ICdwcm9jZXNzTGF0ZW5jeScsIGxhYmVsOiAnUHJvY2VzcyBMYXRlbmN5OiAnIH0sXG4gIHsgdmFsdWU6ICdwcm9jZXNzTmFtZScsIGxhYmVsOiAnUHJvY2VzcyBOYW1lOiAnIH0sXG4gIHtcbiAgICB2YWx1ZTogJ3Byb2Nlc3NTdGFydFRpbWVTdGFtcCcsXG4gICAgbGFiZWw6ICdQcm9jZXNzIFN0YXJ0IFRpbWUsIFVUQyB0aW1lOiAnLFxuICAgIHR5cGU6ICd0aW1lJyxcbiAgfSxcbiAge1xuICAgIHZhbHVlOiAncHJvY2Vzc1RpbWVTdGFtcCcsXG4gICAgbGFiZWw6ICdQcm9jZXNzIFRpbWUsIFVUQyB0aW1lOiAnLFxuICAgIHR5cGU6ICd0aW1lJyxcbiAgfSxcbiAgeyB2YWx1ZTogJ3Byb2Nlc3NlZEV2ZW50cycsIGxhYmVsOiAnUHJvY2Vzc2VkIEV2ZW50czogJyB9LFxuICB7IHZhbHVlOiAncmVwb3J0U3VtbWFyeScsIGxhYmVsOiAnUmVwb3J0IFN1bW1hcnk6ICcgfSxcbiAge1xuICAgIHZhbHVlOiAncnVuU3RhcnRUaW1lU3RhbXAnLFxuICAgIGxhYmVsOiAnUnVuIHN0YXJ0ZWQsIFVUQyB0aW1lOiAnLFxuICAgIHR5cGU6ICd0aW1lJyxcbiAgfSxcbiAgeyB2YWx1ZTogJ3dvcmtpbmdEaXInLCBsYWJlbDogJ1dvcmtpbmcgZGlyZWN0b3J5OiAnIH0sXG5dO1xuXG5leHBvcnQgY29uc3QgbWFpbl9ydW5faW5mbyA9IFtcbiAgeyB2YWx1ZTogJ2lSdW4nLCBsYWJlbDogJ1J1bjogJyB9LFxuICB7IHZhbHVlOiAnaUx1bWlTZWN0aW9uJywgbGFiZWw6ICdMUyAjOiAnIH0sXG4gIHsgdmFsdWU6ICdpRXZlbnQnLCBsYWJlbDogJ0V2ZW50ICM6ICcgfSxcbiAgeyB2YWx1ZTogJ3J1blN0YXJ0VGltZVN0YW1wJywgbGFiZWw6ICdSdW4gc3RhcnRlZCwgVVRDIHRpbWU6ICcsIHR5cGU6ICd0aW1lJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHJ1bl9pbmZvID0gbWFpbl9ydW5faW5mby5jb25jYXQoYWRkaXRpb25hbF9ydW5faW5mbyk7XG4iXSwic291cmNlUm9vdCI6IiJ9
webpackHotUpdate_N_E("pages/index",{

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9jb25zdGFudHMudHMiXSwibmFtZXMiOlsic2l6ZXMiLCJ0aW55IiwibGFiZWwiLCJzaXplIiwidyIsImgiLCJzbWFsbCIsIm1lZGl1bSIsImxhcmdlIiwiZmlsbCIsImZpZWxkX25hbWUiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwiRk9MREVSU19PUl9QTE9UU19SRURVQ0VSIiwiU0VUX1BMT1RfVE9fT1ZFUkxBWSIsIlNFVF9XSURUSCIsIlNFVF9IRUlHSFQiLCJTRVRfWk9PTUVEX1BMT1RfU0laRSIsIlNFVF9OT1JNQUxJWkUiLCJTRVRfU1RBVFMiLCJTRVRfRVJSX0JBUlMiLCJTSE9XIiwiSlNST09UX01PREUiLCJTRVRfUEFSQU1TX0ZPUl9DVVNUT01JWkUiLCJOQVZfUkVEVUNFUiIsIlNFVF9TRUFSQ0hfQllfREFUQVNFVF9OQU1FIiwiU0VUX1NFQVJDSF9CWV9SVU5fTlVNQkVSIiwiUkVGRVJFTkNFX1JFRENFUiIsIkNIQU5HRV9UUklQTEVTX1ZBTFVFUyIsIk9QRU5fTU9EQUwiLCJvdmVybGF5T3B0aW9ucyIsInZhbHVlIiwieHl6VHlwZXMiLCJ3aXRoUmVmZXJlbmNlIiwiZGF0YVNldFNlbGVjdGlvbnMiLCJ2aWV3UG9zaXRpb25zIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJhZGRpdGlvbmFsX3J1bl9pbmZvIiwidHlwZSIsIm1haW5fcnVuX2luZm8iLCJydW5faW5mbyIsImNvbmNhdCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFPLElBQU1BLEtBQUssR0FBRztBQUNuQkMsTUFBSSxFQUFFO0FBQ0pDLFNBQUssRUFBRSxNQURIO0FBRUpDLFFBQUksRUFBRTtBQUNKQyxPQUFDLEVBQUUsRUFEQztBQUVKQyxPQUFDLEVBQUU7QUFGQztBQUZG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O0FBQUU2QixPQUFLLEVBQUUsZUFBVDtBQUEwQjdCLE9BQUssRUFBRTtBQUFqQyxDQW5CaUMsRUFvQmpDO0FBQ0U2QixPQUFLLEVBQUUsbUJBRFQ7QUFFRTdCLE9BQUssRUFBRSx5QkFGVDtBQUdFb0MsTUFBSSxFQUFFO0FBSFIsQ0FwQmlDLEVBeUJqQztBQUFFUCxPQUFLLEVBQUUsWUFBVDtBQUF1QjdCLE9BQUssRUFBRTtBQUE5QixDQXpCaUMsQ0FBNUI7QUE0QkEsSUFBTXFDLGFBQWEsR0FBRyxDQUMzQjtBQUFFUixPQUFLLEVBQUUsTUFBVDtBQUFpQjdCLE9BQUssRUFBRTtBQUF4QixDQUQyQixFQUUzQjtBQUFFNkIsT0FBSyxFQUFFLGNBQVQ7QUFBeUI3QixPQUFLLEVBQUU7QUFBaEMsQ0FGMkIsRUFHM0I7QUFBRTZCLE9BQUssRUFBRSxRQUFUO0FBQW1CN0IsT0FBSyxFQUFFO0FBQTFCLENBSDJCLEVBSTNCO0FBQUU2QixPQUFLLEVBQUUsbUJBQVQ7QUFBOEI3QixPQUFLLEVBQUUseUJBQXJDO0FBQWdFb0MsTUFBSSxFQUFFO0FBQXRFLENBSjJCLENBQXRCO0FBT0EsSUFBTUUsUUFBUSxHQUFHRCxhQUFhLENBQUNFLE1BQWQsQ0FBcUJKLG1CQUFyQixDQUFqQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5mMWI1NmIzYjY0NjZmMjU4ZmY2OC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiZXhwb3J0IGNvbnN0IHNpemVzID0ge1xuICB0aW55OiB7XG4gICAgbGFiZWw6ICdUaW55JyxcbiAgICBzaXplOiB7XG4gICAgICB3OiA2NyxcbiAgICAgIGg6IDUwLFxuICAgIH0sXG4gIH0sXG4gIHNtYWxsOiB7XG4gICAgbGFiZWw6ICdTbWFsbCcsXG4gICAgc2l6ZToge1xuICAgICAgdzogMTMzLFxuICAgICAgaDogMTAwLFxuICAgIH0sXG4gIH0sXG4gIG1lZGl1bToge1xuICAgIGxhYmVsOiAnTWVkaXVtJyxcbiAgICBzaXplOiB7XG4gICAgICB3OiAyNjYsXG4gICAgICBoOiAyMDAsXG4gICAgfSxcbiAgfSxcbiAgbGFyZ2U6IHtcbiAgICBsYWJlbDogJ0xhcmdlJyxcbiAgICBzaXplOiB7XG4gICAgICB3OiA1MzIsXG4gICAgICBoOiA0MDAsXG4gICAgfSxcbiAgfSxcbiAgZmlsbDoge1xuICAgIGxhYmVsOiAnRmlsbCcsXG4gICAgc2l6ZToge1xuICAgICAgdzogNzIwLFxuICAgICAgaDogNTQxLFxuICAgIH0sXG4gIH0sXG59O1xuXG5leHBvcnQgY29uc3QgZmllbGRfbmFtZTogYW55ID0ge1xuICBkYXRhc2V0X25hbWU6ICdEYXRhc2V0IG5hbWUnLFxuICBydW5fbnVtYmVyOiAnUnVuIG51bWJlcicsXG4gIGxhYmVsOiAnbGFiZWwnLFxufTtcblxuZXhwb3J0IGNvbnN0IEZPTERFUlNfT1JfUExPVFNfUkVEVUNFUiA9IHtcbiAgU0VUX1BMT1RfVE9fT1ZFUkxBWTogJ1NFVF9QTE9UX1RPX09WRVJMQVknLFxuICBTRVRfV0lEVEg6ICdTRVRfV0lEVEgnLFxuICBTRVRfSEVJR0hUOiAnU0VUX0hFSUdIVCcsXG4gIFNFVF9aT09NRURfUExPVF9TSVpFOiAnU0VUX1pPT01FRF9QTE9UX1NJWkUnLFxuICBTRVRfTk9STUFMSVpFOiAnU0VUX05PUk1BTElaRScsXG4gIFNFVF9TVEFUUzogJ1NFVF9TVEFUUycsXG4gIFNFVF9FUlJfQkFSUzogJ1NFVF9FUlJfQkFSUycsXG4gIFNIT1c6ICdTSE9XJyxcbiAgSlNST09UX01PREU6ICdKU1JPT1RfTU9ERScsXG4gIFNFVF9QQVJBTVNfRk9SX0NVU1RPTUlaRTogJ1NFVF9QQVJBTVNfRk9SX0NVU1RPTUlaRScsXG59O1xuXG5leHBvcnQgY29uc3QgTkFWX1JFRFVDRVIgPSB7XG4gIFNFVF9TRUFSQ0hfQllfREFUQVNFVF9OQU1FOiAnU0VUX1NFQVJDSF9CWV9EQVRBU0VUX05BTUUnLFxuICBTRVRfU0VBUkNIX0JZX1JVTl9OVU1CRVI6ICdTRVRfU0VBUkNIX0JZX1JVTl9OVU1CRVInLFxufTtcblxuZXhwb3J0IGNvbnN0IFJFRkVSRU5DRV9SRURDRVIgPSB7XG4gIENIQU5HRV9UUklQTEVTX1ZBTFVFUzogJ0NIQU5HRV9UUklQTEVTX1ZBTFVFUycsXG4gIE9QRU5fTU9EQUw6ICdPUEVOX01PREFMJyxcbn07XG5cbmV4cG9ydCBjb25zdCBvdmVybGF5T3B0aW9ucyA9IFtcbiAgeyBsYWJlbDogJ092ZXJsYXknLCB2YWx1ZTogJ292ZXJsYXknIH0sXG4gIHsgbGFiZWw6ICdPbiBzaWRlJywgdmFsdWU6ICdvblNpZGUnIH0sXG4gIHsgbGFiZWw6ICdPdmVybGF5K3JhdGlvJywgdmFsdWU6ICdyYXRpb292ZXJsYXknIH0sXG4gIHsgbGFiZWw6ICdTdGFja2VkJywgdmFsdWU6ICdzdGFja2VkJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHh5elR5cGVzID0gW1xuICB7IGxhYmVsOiAnRGVmYXVsdCcsIHZhbHVlOiAnJyB9LFxuICB7IGxhYmVsOiAnTGluZWFyJywgdmFsdWU6ICdsaW4nIH0sXG4gIHsgbGFiZWw6ICdMb2cnLCB2YWx1ZTogJ2xvZycgfSxcbl07XG5cbmV4cG9ydCBjb25zdCB3aXRoUmVmZXJlbmNlID0gW1xuICB7IGxhYmVsOiAnRGVmYXVsdCcsIHZhbHVlOiAnJyB9LFxuICB7IGxhYmVsOiAnWWVzJywgdmFsdWU6ICd5ZXMnIH0sXG4gIHsgbGFiZWw6ICdObycsIHZhbHVlOiAnbm8nIH0sXG5dO1xuXG5leHBvcnQgY29uc3QgZGF0YVNldFNlbGVjdGlvbnMgPSBbXG4gIHtcbiAgICBsYWJlbDogJ0RhdGFzZXQgU2VsZWN0JyxcbiAgICB2YWx1ZTogJ2RhdGFzZXRTZWxlY3QnLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdEYXRhc2V0IEJ1aWxkZXInLFxuICAgIHZhbHVlOiAnZGF0YXNldEJ1aWxkZXInLFxuICB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHZpZXdQb3NpdGlvbnMgPSBbXG4gIHsgbGFiZWw6ICdIb3Jpem9udGFsJywgdmFsdWU6ICdob3Jpem9udGFsJyB9LFxuICB7IGxhYmVsOiAnVmVydGljYWwnLCB2YWx1ZTogJ3ZlcnRpY2FsJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zID0gW1xuICB7IGxhYmVsOiAnNTAlIDogNTAlJywgdmFsdWU6ICc1MCUnIH0sXG4gIHsgbGFiZWw6ICcyNSUgOiA3NSUnLCB2YWx1ZTogJzI1JScgfSxcbl07XG5cbmV4cG9ydCBjb25zdCBhZGRpdGlvbmFsX3J1bl9pbmZvID0gW1xuICB7IHZhbHVlOiAnQ01TU1dfVmVyc2lvbicsIGxhYmVsOiAnQ01TU1cgdmVyc2lvbjogJyB9LFxuICB7IHZhbHVlOiAnQ2VydGlmaWNhdGlvblN1bW1hcnknLCBsYWJlbDogJ0NlcnRpZmljYXRpb25TdW1tYXJ5OiAnIH0sXG4gIHsgdmFsdWU6ICdob3N0TmFtZScsIGxhYmVsOiAnSG9zdCBuYW1lOiAnIH0sXG4gIHsgdmFsdWU6ICdpRXZlbnQnLCBsYWJlbDogJ0V2ZW50ICM6ICcgfSxcbiAgeyB2YWx1ZTogJ3Byb2Nlc3NJRCcsIGxhYmVsOiAnUHJvY2VzcyBJRDogJyB9LFxuICB7IHZhbHVlOiAncHJvY2Vzc0xhdGVuY3knLCBsYWJlbDogJ1Byb2Nlc3MgTGF0ZW5jeTogJyB9LFxuICB7IHZhbHVlOiAncHJvY2Vzc05hbWUnLCBsYWJlbDogJ1Byb2Nlc3MgTmFtZTogJyB9LFxuICB7XG4gICAgdmFsdWU6ICdwcm9jZXNzU3RhcnRUaW1lU3RhbXAnLFxuICAgIGxhYmVsOiAnUHJvY2VzcyBTdGFydCBUaW1lLCBVVEMgdGltZTogJyxcbiAgICB0eXBlOiAndGltZScsXG4gIH0sXG4gIHtcbiAgICB2YWx1ZTogJ3Byb2Nlc3NUaW1lU3RhbXAnLFxuICAgIGxhYmVsOiAnUHJvY2VzcyBUaW1lLCBVVEMgdGltZTogJyxcbiAgICB0eXBlOiAndGltZScsXG4gIH0sXG4gIHsgdmFsdWU6ICdwcm9jZXNzZWRFdmVudHMnLCBsYWJlbDogJ1Byb2Nlc3NlZCBFdmVudHM6ICcgfSxcbiAgeyB2YWx1ZTogJ3JlcG9ydFN1bW1hcnknLCBsYWJlbDogJ1JlcG9ydCBTdW1tYXJ5OiAnIH0sXG4gIHtcbiAgICB2YWx1ZTogJ3J1blN0YXJ0VGltZVN0YW1wJyxcbiAgICBsYWJlbDogJ1J1biBzdGFydGVkLCBVVEMgdGltZTogJyxcbiAgICB0eXBlOiAndGltZScsXG4gIH0sXG4gIHsgdmFsdWU6ICd3b3JraW5nRGlyJywgbGFiZWw6ICdXb3JraW5nIGRpcmVjdG9yeTogJyB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IG1haW5fcnVuX2luZm8gPSBbXG4gIHsgdmFsdWU6ICdpUnVuJywgbGFiZWw6ICdSdW46ICcgfSxcbiAgeyB2YWx1ZTogJ2lMdW1pU2VjdGlvbicsIGxhYmVsOiAnTFMgIzogJyB9LFxuICB7IHZhbHVlOiAnaUV2ZW50JywgbGFiZWw6ICdFdmVudCAjOiAnIH0sXG4gIHsgdmFsdWU6ICdydW5TdGFydFRpbWVTdGFtcCcsIGxhYmVsOiAnUnVuIHN0YXJ0ZWQsIFVUQyB0aW1lOiAnLCB0eXBlOiAndGltZScgfSxcbl07XG5cbmV4cG9ydCBjb25zdCBydW5faW5mbyA9IG1haW5fcnVuX2luZm8uY29uY2F0KGFkZGl0aW9uYWxfcnVuX2luZm8pO1xuIl0sInNvdXJjZVJvb3QiOiIifQ==
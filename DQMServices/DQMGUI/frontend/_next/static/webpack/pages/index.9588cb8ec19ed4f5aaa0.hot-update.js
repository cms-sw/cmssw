webpackHotUpdate_N_E("pages/index",{

/***/ "./components/navigation/liveModeHeader.tsx":
/*!**************************************************!*\
  !*** ./components/navigation/liveModeHeader.tsx ***!
  \**************************************************/
/*! exports provided: LiveModeHeader */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LiveModeHeader", function() { return LiveModeHeader; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/navigation/liveModeHeader.tsx",
    _this = undefined,
    _s2 = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];










var Title = antd__WEBPACK_IMPORTED_MODULE_1__["Typography"].Title;
var LiveModeHeader = function LiveModeHeader(_ref) {
  _s2();

  var _s = $RefreshSig$();

  var query = _ref.query;
  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__["store"]);
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["CustomForm"], {
    display: "flex",
    style: {
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 27,
      columnNumber: 7
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_6__["main_run_info"].map(_s(function (info) {
    _s();

    var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_4__["FormatParamsForAPI"])(globalState, query, info.value, 'HLT/EventInfo');

    var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_8__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number, not_older_than]),
        data = _useRequest.data,
        isLoading = _useRequest.isLoading;

    return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["CutomFormItem"], {
      space: "8",
      width: "fit-content",
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.common.white,
      name: info.label,
      label: info.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 41,
        columnNumber: 13
      }
    }, __jsx(Title, {
      level: 4,
      style: {
        display: 'contents',
        color: "".concat(update ? _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.notification.error)
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 48,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 58,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_9__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"]];
  }))));
};

_s2(LiveModeHeader, "amC1/c9ldnJBSldn3lb055gydI4=");

_c = LiveModeHeader;

var _c;

$RefreshReg$(_c, "LiveModeHeader");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwiZ2xvYmFsU3RhdGUiLCJSZWFjdCIsInN0b3JlIiwiYWxpZ25JdGVtcyIsIm1haW5fcnVuX2luZm8iLCJtYXAiLCJpbmZvIiwicGFyYW1zX2Zvcl9hcGkiLCJGb3JtYXRQYXJhbXNGb3JBUEkiLCJ2YWx1ZSIsInVzZVJlcXVlc3QiLCJnZXRfanJvb3RfcGxvdCIsImRhdGFzZXRfbmFtZSIsInJ1bl9udW1iZXIiLCJub3Rfb2xkZXJfdGhhbiIsImRhdGEiLCJpc0xvYWRpbmciLCJ0aGVtZSIsImNvbG9ycyIsImNvbW1vbiIsIndoaXRlIiwibGFiZWwiLCJkaXNwbGF5IiwiY29sb3IiLCJ1cGRhdGUiLCJub3RpZmljYXRpb24iLCJzdWNjZXNzIiwiZXJyb3IiLCJnZXRfbGFiZWwiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFJQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtJQUNRQSxLLEdBQVVDLCtDLENBQVZELEs7QUFNRCxJQUFNRSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQW9DO0FBQUE7O0FBQUE7O0FBQUEsTUFBakNDLEtBQWlDLFFBQWpDQSxLQUFpQztBQUNoRSxNQUFNQyxXQUFXLEdBQUdDLGdEQUFBLENBQWlCQywrREFBakIsQ0FBcEI7QUFFQSxTQUNFLDREQUNFLE1BQUMsNERBQUQ7QUFBWSxXQUFPLEVBQUMsTUFBcEI7QUFBMkIsU0FBSyxFQUFFO0FBQUVDLGdCQUFVLEVBQUU7QUFBZCxLQUFsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dDLHdEQUFhLENBQUNDLEdBQWQsSUFBa0IsVUFBQ0MsSUFBRCxFQUFxQjtBQUFBOztBQUN0QyxRQUFNQyxjQUFjLEdBQUdDLHVGQUFrQixDQUN2Q1IsV0FEdUMsRUFFdkNELEtBRnVDLEVBR3ZDTyxJQUFJLENBQUNHLEtBSGtDLEVBSXZDLGVBSnVDLENBQXpDOztBQURzQyxzQkFPVkMsb0VBQVUsQ0FDcENDLHFFQUFjLENBQUNKLGNBQUQsQ0FEc0IsRUFFcEMsRUFGb0MsRUFHcEMsQ0FBQ1IsS0FBSyxDQUFDYSxZQUFQLEVBQXFCYixLQUFLLENBQUNjLFVBQTNCLEVBQXVDQyxjQUF2QyxDQUhvQyxDQVBBO0FBQUEsUUFPOUJDLElBUDhCLGVBTzlCQSxJQVA4QjtBQUFBLFFBT3hCQyxTQVB3QixlQU94QkEsU0FQd0I7O0FBWXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUVDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsTUFBYixDQUFvQkMsS0FIN0I7QUFJRSxVQUFJLEVBQUVkLElBQUksQ0FBQ2UsS0FKYjtBQUtFLFdBQUssRUFBRWYsSUFBSSxDQUFDZSxLQUxkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FPRSxNQUFDLEtBQUQ7QUFDRSxXQUFLLEVBQUUsQ0FEVDtBQUVFLFdBQUssRUFBRTtBQUNMQyxlQUFPLEVBQUUsVUFESjtBQUVMQyxhQUFLLFlBQUtDLE1BQU0sR0FDWlAsbURBQUssQ0FBQ0MsTUFBTixDQUFhTyxZQUFiLENBQTBCQyxPQURkLEdBRVpULG1EQUFLLENBQUNDLE1BQU4sQ0FBYU8sWUFBYixDQUEwQkUsS0FGekI7QUFGQSxPQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FVR1gsU0FBUyxHQUFHLE1BQUMseUNBQUQ7QUFBTSxVQUFJLEVBQUMsT0FBWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQUgsR0FBMkJZLHdEQUFTLENBQUN0QixJQUFELEVBQU9TLElBQVAsQ0FWaEQsQ0FQRixDQURGO0FBc0JELEdBbENBO0FBQUEsWUFPNkJMLDREQVA3QjtBQUFBLEtBREgsQ0FERixDQURGO0FBeUNELENBNUNNOztJQUFNWixjOztLQUFBQSxjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4Ljk1ODhjYjhlYzE5ZWQ0ZjVhYWEwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IFNwaW4sIFR5cG9ncmFwaHkgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7XHJcbiAgQ3VzdG9tRm9ybSxcclxuICBDdXRvbUZvcm1JdGVtLFxyXG59IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSB9IGZyb20gJy4uL3Bsb3RzL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcywgSW5mb1Byb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBtYWluX3J1bl9pbmZvIH0gZnJvbSAnLi4vY29uc3RhbnRzJztcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnO1xyXG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQgeyBnZXRfbGFiZWwgfSBmcm9tICcuLi91dGlscyc7XHJcbmNvbnN0IHsgVGl0bGUgfSA9IFR5cG9ncmFwaHk7XHJcblxyXG5pbnRlcmZhY2UgTGl2ZU1vZGVIZWFkZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBMaXZlTW9kZUhlYWRlciA9ICh7IHF1ZXJ5IH06IExpdmVNb2RlSGVhZGVyUHJvcHMpID0+IHtcclxuICBjb25zdCBnbG9iYWxTdGF0ZSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPD5cclxuICAgICAgPEN1c3RvbUZvcm0gZGlzcGxheT1cImZsZXhcIiBzdHlsZT17eyBhbGlnbkl0ZW1zOiAnY2VudGVyJywgfX0+XHJcbiAgICAgICAge21haW5fcnVuX2luZm8ubWFwKChpbmZvOiBJbmZvUHJvcHMpID0+IHtcclxuICAgICAgICAgIGNvbnN0IHBhcmFtc19mb3JfYXBpID0gRm9ybWF0UGFyYW1zRm9yQVBJKFxyXG4gICAgICAgICAgICBnbG9iYWxTdGF0ZSxcclxuICAgICAgICAgICAgcXVlcnksXHJcbiAgICAgICAgICAgIGluZm8udmFsdWUsXHJcbiAgICAgICAgICAgICdITFQvRXZlbnRJbmZvJ1xyXG4gICAgICAgICAgKTtcclxuICAgICAgICAgIGNvbnN0IHsgZGF0YSwgaXNMb2FkaW5nIH0gPSB1c2VSZXF1ZXN0KFxyXG4gICAgICAgICAgICBnZXRfanJvb3RfcGxvdChwYXJhbXNfZm9yX2FwaSksXHJcbiAgICAgICAgICAgIHt9LFxyXG4gICAgICAgICAgICBbcXVlcnkuZGF0YXNldF9uYW1lLCBxdWVyeS5ydW5fbnVtYmVyLCBub3Rfb2xkZXJfdGhhbl1cclxuICAgICAgICAgICk7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8Q3V0b21Gb3JtSXRlbVxyXG4gICAgICAgICAgICAgIHNwYWNlPVwiOFwiXHJcbiAgICAgICAgICAgICAgd2lkdGg9XCJmaXQtY29udGVudFwiXHJcbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5jb21tb24ud2hpdGV9XHJcbiAgICAgICAgICAgICAgbmFtZT17aW5mby5sYWJlbH1cclxuICAgICAgICAgICAgICBsYWJlbD17aW5mby5sYWJlbH1cclxuICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgIDxUaXRsZVxyXG4gICAgICAgICAgICAgICAgbGV2ZWw9ezR9XHJcbiAgICAgICAgICAgICAgICBzdHlsZT17e1xyXG4gICAgICAgICAgICAgICAgICBkaXNwbGF5OiAnY29udGVudHMnLFxyXG4gICAgICAgICAgICAgICAgICBjb2xvcjogYCR7dXBkYXRlXHJcbiAgICAgICAgICAgICAgICAgICAgPyB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLnN1Y2Nlc3NcclxuICAgICAgICAgICAgICAgICAgICA6IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uZXJyb3JcclxuICAgICAgICAgICAgICAgICAgICB9YCxcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IDxTcGluIHNpemU9XCJzbWFsbFwiIC8+IDogZ2V0X2xhYmVsKGluZm8sIGRhdGEpfVxyXG4gICAgICAgICAgICAgIDwvVGl0bGU+XHJcbiAgICAgICAgICAgIDwvQ3V0b21Gb3JtSXRlbT5cclxuICAgICAgICAgICk7XHJcbiAgICAgICAgfSl9XHJcbiAgICAgIDwvQ3VzdG9tRm9ybT5cclxuICAgIDwvPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
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
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/navigation/liveModeHeader.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];









var Title = antd__WEBPACK_IMPORTED_MODULE_1__["Typography"].Title;
var LiveModeHeader = function LiveModeHeader(_ref) {
  var _s = $RefreshSig$();

  var query = _ref.query;
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["CustomForm"], {
    display: "flex",
    style: {
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 25,
      columnNumber: 7
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_5__["main_run_info"].map(_s(function (info) {
    _s();

    var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_4__["FormatParamsForAPI"])(globalState, query, info.value, 'HLT/EventInfo');

    var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_6__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_7__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number, not_older_than]),
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
        lineNumber: 39,
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
        lineNumber: 46,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 56,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_8__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_6__["useRequest"]];
  }))));
};
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwiYWxpZ25JdGVtcyIsIm1haW5fcnVuX2luZm8iLCJtYXAiLCJpbmZvIiwicGFyYW1zX2Zvcl9hcGkiLCJGb3JtYXRQYXJhbXNGb3JBUEkiLCJnbG9iYWxTdGF0ZSIsInZhbHVlIiwidXNlUmVxdWVzdCIsImdldF9qcm9vdF9wbG90IiwiZGF0YXNldF9uYW1lIiwicnVuX251bWJlciIsIm5vdF9vbGRlcl90aGFuIiwiZGF0YSIsImlzTG9hZGluZyIsInRoZW1lIiwiY29sb3JzIiwiY29tbW9uIiwid2hpdGUiLCJsYWJlbCIsImRpc3BsYXkiLCJjb2xvciIsInVwZGF0ZSIsIm5vdGlmaWNhdGlvbiIsInN1Y2Nlc3MiLCJlcnJvciIsImdldF9sYWJlbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFJQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7SUFDUUEsSyxHQUFVQywrQyxDQUFWRCxLO0FBTUQsSUFBTUUsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUFvQztBQUFBOztBQUFBLE1BQWpDQyxLQUFpQyxRQUFqQ0EsS0FBaUM7QUFFaEUsU0FDRSw0REFDRSxNQUFDLDREQUFEO0FBQVksV0FBTyxFQUFDLE1BQXBCO0FBQTJCLFNBQUssRUFBRTtBQUFFQyxnQkFBVSxFQUFFO0FBQWQsS0FBbEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHQyx3REFBYSxDQUFDQyxHQUFkLElBQWtCLFVBQUNDLElBQUQsRUFBcUI7QUFBQTs7QUFDdEMsUUFBTUMsY0FBYyxHQUFHQyx1RkFBa0IsQ0FDdkNDLFdBRHVDLEVBRXZDUCxLQUZ1QyxFQUd2Q0ksSUFBSSxDQUFDSSxLQUhrQyxFQUl2QyxlQUp1QyxDQUF6Qzs7QUFEc0Msc0JBT1ZDLG9FQUFVLENBQ3BDQyxxRUFBYyxDQUFDTCxjQUFELENBRHNCLEVBRXBDLEVBRm9DLEVBR3BDLENBQUNMLEtBQUssQ0FBQ1csWUFBUCxFQUFxQlgsS0FBSyxDQUFDWSxVQUEzQixFQUF1Q0MsY0FBdkMsQ0FIb0MsQ0FQQTtBQUFBLFFBTzlCQyxJQVA4QixlQU85QkEsSUFQOEI7QUFBQSxRQU94QkMsU0FQd0IsZUFPeEJBLFNBUHdCOztBQVl0QyxXQUNFLE1BQUMsK0RBQUQ7QUFDRSxXQUFLLEVBQUMsR0FEUjtBQUVFLFdBQUssRUFBQyxhQUZSO0FBR0UsV0FBSyxFQUFFQyxtREFBSyxDQUFDQyxNQUFOLENBQWFDLE1BQWIsQ0FBb0JDLEtBSDdCO0FBSUUsVUFBSSxFQUFFZixJQUFJLENBQUNnQixLQUpiO0FBS0UsV0FBSyxFQUFFaEIsSUFBSSxDQUFDZ0IsS0FMZDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BT0UsTUFBQyxLQUFEO0FBQ0UsV0FBSyxFQUFFLENBRFQ7QUFFRSxXQUFLLEVBQUU7QUFDTEMsZUFBTyxFQUFFLFVBREo7QUFFTEMsYUFBSyxZQUFLQyxNQUFNLEdBQ1pQLG1EQUFLLENBQUNDLE1BQU4sQ0FBYU8sWUFBYixDQUEwQkMsT0FEZCxHQUVaVCxtREFBSyxDQUFDQyxNQUFOLENBQWFPLFlBQWIsQ0FBMEJFLEtBRnpCO0FBRkEsT0FGVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BVUdYLFNBQVMsR0FBRyxNQUFDLHlDQUFEO0FBQU0sVUFBSSxFQUFDLE9BQVg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUFILEdBQTJCWSx3REFBUyxDQUFDdkIsSUFBRCxFQUFPVSxJQUFQLENBVmhELENBUEYsQ0FERjtBQXNCRCxHQWxDQTtBQUFBLFlBTzZCTCw0REFQN0I7QUFBQSxLQURILENBREYsQ0FERjtBQXlDRCxDQTNDTTtLQUFNVixjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmViMTRjNmI1ZjU2NGNhZTEzMmE5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IFNwaW4sIFR5cG9ncmFwaHkgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7XHJcbiAgQ3VzdG9tRm9ybSxcclxuICBDdXRvbUZvcm1JdGVtLFxyXG59IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSB9IGZyb20gJy4uL3Bsb3RzL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMsIEluZm9Qcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgbWFpbl9ydW5faW5mbyB9IGZyb20gJy4uL2NvbnN0YW50cyc7XHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0JztcclxuaW1wb3J0IHsgZ2V0X2pyb290X3Bsb3QgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHsgZ2V0X2xhYmVsIH0gZnJvbSAnLi4vdXRpbHMnO1xyXG5jb25zdCB7IFRpdGxlIH0gPSBUeXBvZ3JhcGh5O1xyXG5cclxuaW50ZXJmYWNlIExpdmVNb2RlSGVhZGVyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgTGl2ZU1vZGVIZWFkZXIgPSAoeyBxdWVyeSB9OiBMaXZlTW9kZUhlYWRlclByb3BzKSA9PiB7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8PlxyXG4gICAgICA8Q3VzdG9tRm9ybSBkaXNwbGF5PVwiZmxleFwiIHN0eWxlPXt7IGFsaWduSXRlbXM6ICdjZW50ZXInLCB9fT5cclxuICAgICAgICB7bWFpbl9ydW5faW5mby5tYXAoKGluZm86IEluZm9Qcm9wcykgPT4ge1xyXG4gICAgICAgICAgY29uc3QgcGFyYW1zX2Zvcl9hcGkgPSBGb3JtYXRQYXJhbXNGb3JBUEkoXHJcbiAgICAgICAgICAgIGdsb2JhbFN0YXRlLFxyXG4gICAgICAgICAgICBxdWVyeSxcclxuICAgICAgICAgICAgaW5mby52YWx1ZSxcclxuICAgICAgICAgICAgJ0hMVC9FdmVudEluZm8nXHJcbiAgICAgICAgICApO1xyXG4gICAgICAgICAgY29uc3QgeyBkYXRhLCBpc0xvYWRpbmcgfSA9IHVzZVJlcXVlc3QoXHJcbiAgICAgICAgICAgIGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSxcclxuICAgICAgICAgICAge30sXHJcbiAgICAgICAgICAgIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXIsIG5vdF9vbGRlcl90aGFuXVxyXG4gICAgICAgICAgKTtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxDdXRvbUZvcm1JdGVtXHJcbiAgICAgICAgICAgICAgc3BhY2U9XCI4XCJcclxuICAgICAgICAgICAgICB3aWR0aD1cImZpdC1jb250ZW50XCJcclxuICAgICAgICAgICAgICBjb2xvcj17dGhlbWUuY29sb3JzLmNvbW1vbi53aGl0ZX1cclxuICAgICAgICAgICAgICBuYW1lPXtpbmZvLmxhYmVsfVxyXG4gICAgICAgICAgICAgIGxhYmVsPXtpbmZvLmxhYmVsfVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPFRpdGxlXHJcbiAgICAgICAgICAgICAgICBsZXZlbD17NH1cclxuICAgICAgICAgICAgICAgIHN0eWxlPXt7XHJcbiAgICAgICAgICAgICAgICAgIGRpc3BsYXk6ICdjb250ZW50cycsXHJcbiAgICAgICAgICAgICAgICAgIGNvbG9yOiBgJHt1cGRhdGVcclxuICAgICAgICAgICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xyXG4gICAgICAgICAgICAgICAgICAgIDogdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5lcnJvclxyXG4gICAgICAgICAgICAgICAgICAgIH1gLFxyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gPFNwaW4gc2l6ZT1cInNtYWxsXCIgLz4gOiBnZXRfbGFiZWwoaW5mbywgZGF0YSl9XHJcbiAgICAgICAgICAgICAgPC9UaXRsZT5cclxuICAgICAgICAgICAgPC9DdXRvbUZvcm1JdGVtPlxyXG4gICAgICAgICAgKTtcclxuICAgICAgICB9KX1cclxuICAgICAgPC9DdXN0b21Gb3JtPlxyXG4gICAgPC8+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
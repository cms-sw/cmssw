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
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/navigation/liveModeHeader.tsx",
    _this = undefined,
    _s2 = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];











var Title = antd__WEBPACK_IMPORTED_MODULE_1__["Typography"].Title;
var LiveModeHeader = function LiveModeHeader(_ref) {
  _s2();

  var _s = $RefreshSig$();

  var query = _ref.query;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__["useUpdateLiveMode"])(),
      update = _useUpdateLiveMode.update,
      set_update = _useUpdateLiveMode.set_update,
      not_older_than = _useUpdateLiveMode.not_older_than;

  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__["store"]);
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["CustomForm"], {
    display: "flex",
    style: {
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_7__["main_run_info"].map(_s(function (info) {
    _s();

    var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_5__["FormatParamsForAPI"])(globalState, query, info.value, 'HLT/EventInfo');

    var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_9__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number, not_older_than]),
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
        lineNumber: 47,
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
        lineNumber: 54,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 64,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_10__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
  }))));
};

_s2(LiveModeHeader, "ohC5a37T9gYw9m5ORyIlJoPiizQ=", false, function () {
  return [_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__["useUpdateLiveMode"]];
});

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwibm90X29sZGVyX3RoYW4iLCJnbG9iYWxTdGF0ZSIsIlJlYWN0Iiwic3RvcmUiLCJhbGlnbkl0ZW1zIiwibWFpbl9ydW5faW5mbyIsIm1hcCIsImluZm8iLCJwYXJhbXNfZm9yX2FwaSIsIkZvcm1hdFBhcmFtc0ZvckFQSSIsInZhbHVlIiwidXNlUmVxdWVzdCIsImdldF9qcm9vdF9wbG90IiwiZGF0YXNldF9uYW1lIiwicnVuX251bWJlciIsImRhdGEiLCJpc0xvYWRpbmciLCJ0aGVtZSIsImNvbG9ycyIsImNvbW1vbiIsIndoaXRlIiwibGFiZWwiLCJkaXNwbGF5IiwiY29sb3IiLCJub3RpZmljYXRpb24iLCJzdWNjZXNzIiwiZXJyb3IiLCJnZXRfbGFiZWwiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUdBO0FBTUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtJQUNRQSxLLEdBQVVDLCtDLENBQVZELEs7QUFNRCxJQUFNRSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQW9DO0FBQUE7O0FBQUE7O0FBQUEsTUFBakNDLEtBQWlDLFFBQWpDQSxLQUFpQzs7QUFBQSwyQkFDakJDLG9GQUFpQixFQURBO0FBQUEsTUFDeERDLE1BRHdELHNCQUN4REEsTUFEd0Q7QUFBQSxNQUNoREMsVUFEZ0Qsc0JBQ2hEQSxVQURnRDtBQUFBLE1BQ3BDQyxjQURvQyxzQkFDcENBLGNBRG9DOztBQUVoRSxNQUFNQyxXQUFXLEdBQUdDLGdEQUFBLENBQWlCQywrREFBakIsQ0FBcEI7QUFFQSxTQUNFLDREQUNFLE1BQUMsNERBQUQ7QUFBWSxXQUFPLEVBQUMsTUFBcEI7QUFBMkIsU0FBSyxFQUFFO0FBQUVDLGdCQUFVLEVBQUU7QUFBZCxLQUFsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dDLHdEQUFhLENBQUNDLEdBQWQsSUFBa0IsVUFBQ0MsSUFBRCxFQUFxQjtBQUFBOztBQUN0QyxRQUFNQyxjQUFjLEdBQUdDLHVGQUFrQixDQUN2Q1IsV0FEdUMsRUFFdkNMLEtBRnVDLEVBR3ZDVyxJQUFJLENBQUNHLEtBSGtDLEVBSXZDLGVBSnVDLENBQXpDOztBQURzQyxzQkFRVkMsb0VBQVUsQ0FDcENDLHFFQUFjLENBQUNKLGNBQUQsQ0FEc0IsRUFFcEMsRUFGb0MsRUFHcEMsQ0FBQ1osS0FBSyxDQUFDaUIsWUFBUCxFQUFxQmpCLEtBQUssQ0FBQ2tCLFVBQTNCLEVBQXVDZCxjQUF2QyxDQUhvQyxDQVJBO0FBQUEsUUFROUJlLElBUjhCLGVBUTlCQSxJQVI4QjtBQUFBLFFBUXhCQyxTQVJ3QixlQVF4QkEsU0FSd0I7O0FBYXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUVDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsTUFBYixDQUFvQkMsS0FIN0I7QUFJRSxVQUFJLEVBQUViLElBQUksQ0FBQ2MsS0FKYjtBQUtFLFdBQUssRUFBRWQsSUFBSSxDQUFDYyxLQUxkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FPRSxNQUFDLEtBQUQ7QUFDRSxXQUFLLEVBQUUsQ0FEVDtBQUVFLFdBQUssRUFBRTtBQUNMQyxlQUFPLEVBQUUsVUFESjtBQUVMQyxhQUFLLFlBQUt6QixNQUFNLEdBQ1ZtQixtREFBSyxDQUFDQyxNQUFOLENBQWFNLFlBQWIsQ0FBMEJDLE9BRGhCLEdBRVZSLG1EQUFLLENBQUNDLE1BQU4sQ0FBYU0sWUFBYixDQUEwQkUsS0FGM0I7QUFGQSxPQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FVR1YsU0FBUyxHQUFHLE1BQUMseUNBQUQ7QUFBTSxVQUFJLEVBQUMsT0FBWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQUgsR0FBMkJXLHlEQUFTLENBQUNwQixJQUFELEVBQU9RLElBQVAsQ0FWaEQsQ0FQRixDQURGO0FBc0JELEdBbkNBO0FBQUEsWUFRNkJKLDREQVI3QjtBQUFBLEtBREgsQ0FERixDQURGO0FBbUVELENBdkVNOztJQUFNaEIsYztVQUNvQ0UsNEU7OztLQURwQ0YsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5lMWYxZDk1YzY3NGVlMzdiYzJlMy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBCdXR0b24sIFRvb2x0aXAsIFNwaW4sIFR5cG9ncmFwaHkgfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgUGF1c2VPdXRsaW5lZCwgUGxheUNpcmNsZU91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHtcclxuICBDdXN0b21Db2wsXHJcbiAgQ3VzdG9tRGl2LFxyXG4gIEN1c3RvbUZvcm0sXHJcbiAgQ3V0b21Gb3JtSXRlbSxcclxufSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi9zdHlsZXMvdGhlbWUnO1xyXG5pbXBvcnQgeyB1c2VVcGRhdGVMaXZlTW9kZSB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVVwZGF0ZUluTGl2ZU1vZGUnO1xyXG5pbXBvcnQgeyBGb3JtYXRQYXJhbXNGb3JBUEkgfSBmcm9tICcuLi9wbG90cy9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMsIEluZm9Qcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgbWFpbl9ydW5faW5mbyB9IGZyb20gJy4uL2NvbnN0YW50cyc7XHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0JztcclxuaW1wb3J0IHsgZ2V0X2pyb290X3Bsb3QgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHsgZ2V0X2xhYmVsIH0gZnJvbSAnLi4vdXRpbHMnO1xyXG5jb25zdCB7IFRpdGxlIH0gPSBUeXBvZ3JhcGh5O1xyXG5cclxuaW50ZXJmYWNlIExpdmVNb2RlSGVhZGVyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgTGl2ZU1vZGVIZWFkZXIgPSAoeyBxdWVyeSB9OiBMaXZlTW9kZUhlYWRlclByb3BzKSA9PiB7XHJcbiAgY29uc3QgeyB1cGRhdGUsIHNldF91cGRhdGUsIG5vdF9vbGRlcl90aGFuIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xyXG4gIGNvbnN0IGdsb2JhbFN0YXRlID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8PlxyXG4gICAgICA8Q3VzdG9tRm9ybSBkaXNwbGF5PVwiZmxleFwiIHN0eWxlPXt7IGFsaWduSXRlbXM6ICdjZW50ZXInLCB9fT5cclxuICAgICAgICB7bWFpbl9ydW5faW5mby5tYXAoKGluZm86IEluZm9Qcm9wcykgPT4ge1xyXG4gICAgICAgICAgY29uc3QgcGFyYW1zX2Zvcl9hcGkgPSBGb3JtYXRQYXJhbXNGb3JBUEkoXHJcbiAgICAgICAgICAgIGdsb2JhbFN0YXRlLFxyXG4gICAgICAgICAgICBxdWVyeSxcclxuICAgICAgICAgICAgaW5mby52YWx1ZSxcclxuICAgICAgICAgICAgJ0hMVC9FdmVudEluZm8nXHJcbiAgICAgICAgICApO1xyXG5cclxuICAgICAgICAgIGNvbnN0IHsgZGF0YSwgaXNMb2FkaW5nIH0gPSB1c2VSZXF1ZXN0KFxyXG4gICAgICAgICAgICBnZXRfanJvb3RfcGxvdChwYXJhbXNfZm9yX2FwaSksXHJcbiAgICAgICAgICAgIHt9LFxyXG4gICAgICAgICAgICBbcXVlcnkuZGF0YXNldF9uYW1lLCBxdWVyeS5ydW5fbnVtYmVyLCBub3Rfb2xkZXJfdGhhbl1cclxuICAgICAgICAgICk7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8Q3V0b21Gb3JtSXRlbVxyXG4gICAgICAgICAgICAgIHNwYWNlPVwiOFwiXHJcbiAgICAgICAgICAgICAgd2lkdGg9XCJmaXQtY29udGVudFwiXHJcbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5jb21tb24ud2hpdGV9XHJcbiAgICAgICAgICAgICAgbmFtZT17aW5mby5sYWJlbH1cclxuICAgICAgICAgICAgICBsYWJlbD17aW5mby5sYWJlbH1cclxuICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgIDxUaXRsZVxyXG4gICAgICAgICAgICAgICAgbGV2ZWw9ezR9XHJcbiAgICAgICAgICAgICAgICBzdHlsZT17e1xyXG4gICAgICAgICAgICAgICAgICBkaXNwbGF5OiAnY29udGVudHMnLFxyXG4gICAgICAgICAgICAgICAgICBjb2xvcjogYCR7dXBkYXRlXHJcbiAgICAgICAgICAgICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xyXG4gICAgICAgICAgICAgICAgICAgICAgOiB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLmVycm9yXHJcbiAgICAgICAgICAgICAgICAgICAgfWAsXHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtpc0xvYWRpbmcgPyA8U3BpbiBzaXplPVwic21hbGxcIiAvPiA6IGdldF9sYWJlbChpbmZvLCBkYXRhKX1cclxuICAgICAgICAgICAgICA8L1RpdGxlPlxyXG4gICAgICAgICAgICA8L0N1dG9tRm9ybUl0ZW0+XHJcbiAgICAgICAgICApO1xyXG4gICAgICAgIH0pfVxyXG4gICAgICA8L0N1c3RvbUZvcm0+XHJcbiAgICAgIHsvKiA8Q3VzdG9tQ29sXHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJmbGV4LWVuZFwiXHJcbiAgICAgICAgZGlzcGxheT1cImZsZXhcIlxyXG4gICAgICAgIGFsaWduaXRlbXM9XCJjZW50ZXJcIlxyXG4gICAgICAgIHRleHR0cmFuc2Zvcm09XCJ1cHBlcmNhc2VcIlxyXG4gICAgICAgIGNvbG9yPXtcclxuICAgICAgICAgIHVwZGF0ZVxyXG4gICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xyXG4gICAgICAgICAgICA6IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uZXJyb3JcclxuICAgICAgICB9XHJcbiAgICAgID5cclxuICAgICAgICBMaXZlIE1vZGVcclxuICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPlxyXG4gICAgICAgICAgPFRvb2x0aXAgdGl0bGU9e2BVcGRhdGluZyBtb2RlIGlzICR7dXBkYXRlID8gJ29uJyA6ICdvZmYnfWB9PlxyXG4gICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgdHlwZT1cInByaW1hcnlcIlxyXG4gICAgICAgICAgICAgIHNoYXBlPVwiY2lyY2xlXCJcclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBzZXRfdXBkYXRlKCF1cGRhdGUpO1xyXG4gICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgaWNvbj17dXBkYXRlID8gPFBhdXNlT3V0bGluZWQgLz4gOiA8UGxheUNpcmNsZU91dGxpbmVkIC8+fVxyXG4gICAgICAgICAgICA+PC9CdXR0b24+XHJcbiAgICAgICAgICA8L1Rvb2x0aXA+XHJcbiAgICAgICAgPC9DdXN0b21EaXY+XHJcbiAgICAgIDwvQ3VzdG9tQ29sPiAqL31cclxuICAgIDwvPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
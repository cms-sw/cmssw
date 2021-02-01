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
        lineNumber: 46,
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
        lineNumber: 53,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 63,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwibm90X29sZGVyX3RoYW4iLCJnbG9iYWxTdGF0ZSIsIlJlYWN0Iiwic3RvcmUiLCJhbGlnbkl0ZW1zIiwibWFpbl9ydW5faW5mbyIsIm1hcCIsImluZm8iLCJwYXJhbXNfZm9yX2FwaSIsIkZvcm1hdFBhcmFtc0ZvckFQSSIsInZhbHVlIiwidXNlUmVxdWVzdCIsImdldF9qcm9vdF9wbG90IiwiZGF0YXNldF9uYW1lIiwicnVuX251bWJlciIsImRhdGEiLCJpc0xvYWRpbmciLCJ0aGVtZSIsImNvbG9ycyIsImNvbW1vbiIsIndoaXRlIiwibGFiZWwiLCJkaXNwbGF5IiwiY29sb3IiLCJub3RpZmljYXRpb24iLCJzdWNjZXNzIiwiZXJyb3IiLCJnZXRfbGFiZWwiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUdBO0FBTUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtJQUNRQSxLLEdBQVVDLCtDLENBQVZELEs7QUFNRCxJQUFNRSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQW9DO0FBQUE7O0FBQUE7O0FBQUEsTUFBakNDLEtBQWlDLFFBQWpDQSxLQUFpQzs7QUFBQSwyQkFDakJDLG9GQUFpQixFQURBO0FBQUEsTUFDeERDLE1BRHdELHNCQUN4REEsTUFEd0Q7QUFBQSxNQUNoREMsVUFEZ0Qsc0JBQ2hEQSxVQURnRDtBQUFBLE1BQ3BDQyxjQURvQyxzQkFDcENBLGNBRG9DOztBQUVoRSxNQUFNQyxXQUFXLEdBQUdDLGdEQUFBLENBQWlCQywrREFBakIsQ0FBcEI7QUFFQSxTQUNFLDREQUNFLE1BQUMsNERBQUQ7QUFBWSxXQUFPLEVBQUMsTUFBcEI7QUFBMkIsU0FBSyxFQUFFO0FBQUVDLGdCQUFVLEVBQUU7QUFBZCxLQUFsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dDLHdEQUFhLENBQUNDLEdBQWQsSUFBa0IsVUFBQ0MsSUFBRCxFQUFxQjtBQUFBOztBQUN0QyxRQUFNQyxjQUFjLEdBQUdDLHVGQUFrQixDQUN2Q1IsV0FEdUMsRUFFdkNMLEtBRnVDLEVBR3ZDVyxJQUFJLENBQUNHLEtBSGtDLEVBSXZDLGVBSnVDLENBQXpDOztBQURzQyxzQkFPVkMsb0VBQVUsQ0FDcENDLHFFQUFjLENBQUNKLGNBQUQsQ0FEc0IsRUFFcEMsRUFGb0MsRUFHcEMsQ0FBQ1osS0FBSyxDQUFDaUIsWUFBUCxFQUFxQmpCLEtBQUssQ0FBQ2tCLFVBQTNCLEVBQXVDZCxjQUF2QyxDQUhvQyxDQVBBO0FBQUEsUUFPOUJlLElBUDhCLGVBTzlCQSxJQVA4QjtBQUFBLFFBT3hCQyxTQVB3QixlQU94QkEsU0FQd0I7O0FBWXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUVDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsTUFBYixDQUFvQkMsS0FIN0I7QUFJRSxVQUFJLEVBQUViLElBQUksQ0FBQ2MsS0FKYjtBQUtFLFdBQUssRUFBRWQsSUFBSSxDQUFDYyxLQUxkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FPRSxNQUFDLEtBQUQ7QUFDRSxXQUFLLEVBQUUsQ0FEVDtBQUVFLFdBQUssRUFBRTtBQUNMQyxlQUFPLEVBQUUsVUFESjtBQUVMQyxhQUFLLFlBQUt6QixNQUFNLEdBQ1ptQixtREFBSyxDQUFDQyxNQUFOLENBQWFNLFlBQWIsQ0FBMEJDLE9BRGQsR0FFWlIsbURBQUssQ0FBQ0MsTUFBTixDQUFhTSxZQUFiLENBQTBCRSxLQUZ6QjtBQUZBLE9BRlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQVVHVixTQUFTLEdBQUcsTUFBQyx5Q0FBRDtBQUFNLFVBQUksRUFBQyxPQUFYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFBSCxHQUEyQlcseURBQVMsQ0FBQ3BCLElBQUQsRUFBT1EsSUFBUCxDQVZoRCxDQVBGLENBREY7QUFzQkQsR0FsQ0E7QUFBQSxZQU82QkosNERBUDdCO0FBQUEsS0FESCxDQURGLENBREY7QUF5Q0QsQ0E3Q007O0lBQU1oQixjO1VBQ29DRSw0RTs7O0tBRHBDRixjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjUxN2UwNjdiMmY5ZjQ5Mzk4ZjlhLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IEJ1dHRvbiwgVG9vbHRpcCwgU3BpbiwgVHlwb2dyYXBoeSB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBQYXVzZU91dGxpbmVkLCBQbGF5Q2lyY2xlT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQge1xyXG4gIEN1c3RvbUNvbCxcclxuICBDdXN0b21EaXYsXHJcbiAgQ3VzdG9tRm9ybSxcclxuICBDdXRvbUZvcm1JdGVtLFxyXG59IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XHJcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSB9IGZyb20gJy4uL3Bsb3RzL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcywgSW5mb1Byb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBtYWluX3J1bl9pbmZvIH0gZnJvbSAnLi4vY29uc3RhbnRzJztcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnO1xyXG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQgeyBnZXRfbGFiZWwgfSBmcm9tICcuLi91dGlscyc7XHJcbmNvbnN0IHsgVGl0bGUgfSA9IFR5cG9ncmFwaHk7XHJcblxyXG5pbnRlcmZhY2UgTGl2ZU1vZGVIZWFkZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBMaXZlTW9kZUhlYWRlciA9ICh7IHF1ZXJ5IH06IExpdmVNb2RlSGVhZGVyUHJvcHMpID0+IHtcclxuICBjb25zdCB7IHVwZGF0ZSwgc2V0X3VwZGF0ZSwgbm90X29sZGVyX3RoYW4gfSA9IHVzZVVwZGF0ZUxpdmVNb2RlKCk7XHJcbiAgY29uc3QgZ2xvYmFsU3RhdGUgPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDw+XHJcbiAgICAgIDxDdXN0b21Gb3JtIGRpc3BsYXk9XCJmbGV4XCIgc3R5bGU9e3sgYWxpZ25JdGVtczogJ2NlbnRlcicsIH19PlxyXG4gICAgICAgIHttYWluX3J1bl9pbmZvLm1hcCgoaW5mbzogSW5mb1Byb3BzKSA9PiB7XHJcbiAgICAgICAgICBjb25zdCBwYXJhbXNfZm9yX2FwaSA9IEZvcm1hdFBhcmFtc0ZvckFQSShcclxuICAgICAgICAgICAgZ2xvYmFsU3RhdGUsXHJcbiAgICAgICAgICAgIHF1ZXJ5LFxyXG4gICAgICAgICAgICBpbmZvLnZhbHVlLFxyXG4gICAgICAgICAgICAnSExUL0V2ZW50SW5mbydcclxuICAgICAgICAgICk7XHJcbiAgICAgICAgICBjb25zdCB7IGRhdGEsIGlzTG9hZGluZyB9ID0gdXNlUmVxdWVzdChcclxuICAgICAgICAgICAgZ2V0X2pyb290X3Bsb3QocGFyYW1zX2Zvcl9hcGkpLFxyXG4gICAgICAgICAgICB7fSxcclxuICAgICAgICAgICAgW3F1ZXJ5LmRhdGFzZXRfbmFtZSwgcXVlcnkucnVuX251bWJlciwgbm90X29sZGVyX3RoYW5dXHJcbiAgICAgICAgICApO1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPEN1dG9tRm9ybUl0ZW1cclxuICAgICAgICAgICAgICBzcGFjZT1cIjhcIlxyXG4gICAgICAgICAgICAgIHdpZHRoPVwiZml0LWNvbnRlbnRcIlxyXG4gICAgICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuY29tbW9uLndoaXRlfVxyXG4gICAgICAgICAgICAgIG5hbWU9e2luZm8ubGFiZWx9XHJcbiAgICAgICAgICAgICAgbGFiZWw9e2luZm8ubGFiZWx9XHJcbiAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICA8VGl0bGVcclxuICAgICAgICAgICAgICAgIGxldmVsPXs0fVxyXG4gICAgICAgICAgICAgICAgc3R5bGU9e3tcclxuICAgICAgICAgICAgICAgICAgZGlzcGxheTogJ2NvbnRlbnRzJyxcclxuICAgICAgICAgICAgICAgICAgY29sb3I6IGAke3VwZGF0ZVxyXG4gICAgICAgICAgICAgICAgICAgID8gdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5zdWNjZXNzXHJcbiAgICAgICAgICAgICAgICAgICAgOiB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLmVycm9yXHJcbiAgICAgICAgICAgICAgICAgICAgfWAsXHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtpc0xvYWRpbmcgPyA8U3BpbiBzaXplPVwic21hbGxcIiAvPiA6IGdldF9sYWJlbChpbmZvLCBkYXRhKX1cclxuICAgICAgICAgICAgICA8L1RpdGxlPlxyXG4gICAgICAgICAgICA8L0N1dG9tRm9ybUl0ZW0+XHJcbiAgICAgICAgICApO1xyXG4gICAgICAgIH0pfVxyXG4gICAgICA8L0N1c3RvbUZvcm0+XHJcbiAgICA8Lz5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
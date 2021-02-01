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
      lineNumber: 29,
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
        lineNumber: 43,
        columnNumber: 13
      }
    }, __jsx(Title, {
      level: 4,
      copyable: true,
      style: {
        color: 'white'
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 50,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 55,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_10__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
  }))));
};

_s2(LiveModeHeader, "3hT9752yTO1zm67y5xCUkGR5kkY=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJub3Rfb2xkZXJfdGhhbiIsImdsb2JhbFN0YXRlIiwiUmVhY3QiLCJzdG9yZSIsImFsaWduSXRlbXMiLCJtYWluX3J1bl9pbmZvIiwibWFwIiwiaW5mbyIsInBhcmFtc19mb3JfYXBpIiwiRm9ybWF0UGFyYW1zRm9yQVBJIiwidmFsdWUiLCJ1c2VSZXF1ZXN0IiwiZ2V0X2pyb290X3Bsb3QiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwiZGF0YSIsImlzTG9hZGluZyIsInRoZW1lIiwiY29sb3JzIiwiY29tbW9uIiwid2hpdGUiLCJsYWJlbCIsImNvbG9yIiwiZ2V0X2xhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUlBO0FBQ0E7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7SUFDUUEsSyxHQUFVQywrQyxDQUFWRCxLO0FBTUQsSUFBTUUsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUFvQztBQUFBOztBQUFBOztBQUFBLE1BQWpDQyxLQUFpQyxRQUFqQ0EsS0FBaUM7O0FBQUEsMkJBQ3JDQyxvRkFBaUIsRUFEb0I7QUFBQSxNQUN4REMsY0FEd0Qsc0JBQ3hEQSxjQUR3RDs7QUFFaEUsTUFBTUMsV0FBVyxHQUFHQyxnREFBQSxDQUFpQkMsK0RBQWpCLENBQXBCO0FBRUEsU0FDRSw0REFDRSxNQUFDLDREQUFEO0FBQVksV0FBTyxFQUFDLE1BQXBCO0FBQTJCLFNBQUssRUFBRTtBQUFFQyxnQkFBVSxFQUFFO0FBQWQsS0FBbEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHQyx3REFBYSxDQUFDQyxHQUFkLElBQWtCLFVBQUNDLElBQUQsRUFBcUI7QUFBQTs7QUFDdEMsUUFBTUMsY0FBYyxHQUFHQyx1RkFBa0IsQ0FDdkNSLFdBRHVDLEVBRXZDSCxLQUZ1QyxFQUd2Q1MsSUFBSSxDQUFDRyxLQUhrQyxFQUl2QyxlQUp1QyxDQUF6Qzs7QUFEc0Msc0JBT1ZDLG9FQUFVLENBQ3BDQyxxRUFBYyxDQUFDSixjQUFELENBRHNCLEVBRXBDLEVBRm9DLEVBR3BDLENBQUNWLEtBQUssQ0FBQ2UsWUFBUCxFQUFxQmYsS0FBSyxDQUFDZ0IsVUFBM0IsRUFBdUNkLGNBQXZDLENBSG9DLENBUEE7QUFBQSxRQU85QmUsSUFQOEIsZUFPOUJBLElBUDhCO0FBQUEsUUFPeEJDLFNBUHdCLGVBT3hCQSxTQVB3Qjs7QUFZdEMsV0FDRSxNQUFDLCtEQUFEO0FBQ0UsV0FBSyxFQUFDLEdBRFI7QUFFRSxXQUFLLEVBQUMsYUFGUjtBQUdFLFdBQUssRUFBRUMsbURBQUssQ0FBQ0MsTUFBTixDQUFhQyxNQUFiLENBQW9CQyxLQUg3QjtBQUlFLFVBQUksRUFBRWIsSUFBSSxDQUFDYyxLQUpiO0FBS0UsV0FBSyxFQUFFZCxJQUFJLENBQUNjLEtBTGQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQU9FLE1BQUMsS0FBRDtBQUNFLFdBQUssRUFBRSxDQURUO0FBRUUsY0FBUSxNQUZWO0FBR0UsV0FBSyxFQUFFO0FBQUVDLGFBQUssRUFBRTtBQUFULE9BSFQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUtHTixTQUFTLEdBQUcsTUFBQyx5Q0FBRDtBQUFNLFVBQUksRUFBQyxPQUFYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFBSCxHQUEyQk8seURBQVMsQ0FBQ2hCLElBQUQsRUFBT1EsSUFBUCxDQUxoRCxDQVBGLENBREY7QUFpQkQsR0E3QkE7QUFBQSxZQU82QkosNERBUDdCO0FBQUEsS0FESCxDQURGLENBREY7QUFvQ0QsQ0F4Q007O0lBQU1kLGM7VUFDZ0JFLDRFOzs7S0FEaEJGLGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMmRlZjFkNGI3ZTc1MThiM2RlMGEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgU3BpbiwgVHlwb2dyYXBoeSB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHtcclxuICBDdXN0b21Gb3JtLFxyXG4gIEN1dG9tRm9ybUl0ZW0sXHJcbn0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vc3R5bGVzL3RoZW1lJztcclxuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi8uLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcclxuaW1wb3J0IHsgRm9ybWF0UGFyYW1zRm9yQVBJIH0gZnJvbSAnLi4vcGxvdHMvcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzLCBJbmZvUHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IG1haW5fcnVuX2luZm8gfSBmcm9tICcuLi9jb25zdGFudHMnO1xyXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XHJcbmltcG9ydCB7IGdldF9qcm9vdF9wbG90IH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcbmltcG9ydCB7IGdldF9sYWJlbCB9IGZyb20gJy4uL3V0aWxzJztcclxuY29uc3QgeyBUaXRsZSB9ID0gVHlwb2dyYXBoeTtcclxuXHJcbmludGVyZmFjZSBMaXZlTW9kZUhlYWRlclByb3BzIHtcclxuICBxdWVyeTogUXVlcnlQcm9wcztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IExpdmVNb2RlSGVhZGVyID0gKHsgcXVlcnkgfTogTGl2ZU1vZGVIZWFkZXJQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHsgbm90X29sZGVyX3RoYW4gfSA9IHVzZVVwZGF0ZUxpdmVNb2RlKCk7XHJcbiAgY29uc3QgZ2xvYmFsU3RhdGUgPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDw+XHJcbiAgICAgIDxDdXN0b21Gb3JtIGRpc3BsYXk9XCJmbGV4XCIgc3R5bGU9e3sgYWxpZ25JdGVtczogJ2NlbnRlcicsIH19PlxyXG4gICAgICAgIHttYWluX3J1bl9pbmZvLm1hcCgoaW5mbzogSW5mb1Byb3BzKSA9PiB7XHJcbiAgICAgICAgICBjb25zdCBwYXJhbXNfZm9yX2FwaSA9IEZvcm1hdFBhcmFtc0ZvckFQSShcclxuICAgICAgICAgICAgZ2xvYmFsU3RhdGUsXHJcbiAgICAgICAgICAgIHF1ZXJ5LFxyXG4gICAgICAgICAgICBpbmZvLnZhbHVlLFxyXG4gICAgICAgICAgICAnSExUL0V2ZW50SW5mbydcclxuICAgICAgICAgICk7XHJcbiAgICAgICAgICBjb25zdCB7IGRhdGEsIGlzTG9hZGluZyB9ID0gdXNlUmVxdWVzdChcclxuICAgICAgICAgICAgZ2V0X2pyb290X3Bsb3QocGFyYW1zX2Zvcl9hcGkpLFxyXG4gICAgICAgICAgICB7fSxcclxuICAgICAgICAgICAgW3F1ZXJ5LmRhdGFzZXRfbmFtZSwgcXVlcnkucnVuX251bWJlciwgbm90X29sZGVyX3RoYW5dXHJcbiAgICAgICAgICApO1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPEN1dG9tRm9ybUl0ZW1cclxuICAgICAgICAgICAgICBzcGFjZT1cIjhcIlxyXG4gICAgICAgICAgICAgIHdpZHRoPVwiZml0LWNvbnRlbnRcIlxyXG4gICAgICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuY29tbW9uLndoaXRlfVxyXG4gICAgICAgICAgICAgIG5hbWU9e2luZm8ubGFiZWx9XHJcbiAgICAgICAgICAgICAgbGFiZWw9e2luZm8ubGFiZWx9XHJcbiAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICA8VGl0bGVcclxuICAgICAgICAgICAgICAgIGxldmVsPXs0fVxyXG4gICAgICAgICAgICAgICAgY29weWFibGVcclxuICAgICAgICAgICAgICAgIHN0eWxlPXt7IGNvbG9yOiAnd2hpdGUnIH19XHJcbiAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IDxTcGluIHNpemU9XCJzbWFsbFwiIC8+IDogZ2V0X2xhYmVsKGluZm8sIGRhdGEpfVxyXG4gICAgICAgICAgICAgIDwvVGl0bGU+XHJcbiAgICAgICAgICAgIDwvQ3V0b21Gb3JtSXRlbT5cclxuICAgICAgICAgICk7XHJcbiAgICAgICAgfSl9XHJcbiAgICAgIDwvQ3VzdG9tRm9ybT5cclxuICAgIDwvPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
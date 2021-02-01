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
      style: {
        display: 'contents',
        color: "".concat(update ? _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.notification.error)
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
        lineNumber: 60,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJub3Rfb2xkZXJfdGhhbiIsImdsb2JhbFN0YXRlIiwiUmVhY3QiLCJzdG9yZSIsImFsaWduSXRlbXMiLCJtYWluX3J1bl9pbmZvIiwibWFwIiwiaW5mbyIsInBhcmFtc19mb3JfYXBpIiwiRm9ybWF0UGFyYW1zRm9yQVBJIiwidmFsdWUiLCJ1c2VSZXF1ZXN0IiwiZ2V0X2pyb290X3Bsb3QiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwiZGF0YSIsImlzTG9hZGluZyIsInRoZW1lIiwiY29sb3JzIiwiY29tbW9uIiwid2hpdGUiLCJsYWJlbCIsImRpc3BsYXkiLCJjb2xvciIsInVwZGF0ZSIsIm5vdGlmaWNhdGlvbiIsInN1Y2Nlc3MiLCJlcnJvciIsImdldF9sYWJlbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFJQTtBQUNBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0lBQ1FBLEssR0FBVUMsK0MsQ0FBVkQsSztBQU1ELElBQU1FLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBb0M7QUFBQTs7QUFBQTs7QUFBQSxNQUFqQ0MsS0FBaUMsUUFBakNBLEtBQWlDOztBQUFBLDJCQUNyQ0Msb0ZBQWlCLEVBRG9CO0FBQUEsTUFDeERDLGNBRHdELHNCQUN4REEsY0FEd0Q7O0FBRWhFLE1BQU1DLFdBQVcsR0FBR0MsZ0RBQUEsQ0FBaUJDLCtEQUFqQixDQUFwQjtBQUVBLFNBQ0UsNERBQ0UsTUFBQyw0REFBRDtBQUFZLFdBQU8sRUFBQyxNQUFwQjtBQUEyQixTQUFLLEVBQUU7QUFBRUMsZ0JBQVUsRUFBRTtBQUFkLEtBQWxDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR0Msd0RBQWEsQ0FBQ0MsR0FBZCxJQUFrQixVQUFDQyxJQUFELEVBQXFCO0FBQUE7O0FBQ3RDLFFBQU1DLGNBQWMsR0FBR0MsdUZBQWtCLENBQ3ZDUixXQUR1QyxFQUV2Q0gsS0FGdUMsRUFHdkNTLElBQUksQ0FBQ0csS0FIa0MsRUFJdkMsZUFKdUMsQ0FBekM7O0FBRHNDLHNCQU9WQyxvRUFBVSxDQUNwQ0MscUVBQWMsQ0FBQ0osY0FBRCxDQURzQixFQUVwQyxFQUZvQyxFQUdwQyxDQUFDVixLQUFLLENBQUNlLFlBQVAsRUFBcUJmLEtBQUssQ0FBQ2dCLFVBQTNCLEVBQXVDZCxjQUF2QyxDQUhvQyxDQVBBO0FBQUEsUUFPOUJlLElBUDhCLGVBTzlCQSxJQVA4QjtBQUFBLFFBT3hCQyxTQVB3QixlQU94QkEsU0FQd0I7O0FBWXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUVDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsTUFBYixDQUFvQkMsS0FIN0I7QUFJRSxVQUFJLEVBQUViLElBQUksQ0FBQ2MsS0FKYjtBQUtFLFdBQUssRUFBRWQsSUFBSSxDQUFDYyxLQUxkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FPRSxNQUFDLEtBQUQ7QUFDRSxXQUFLLEVBQUUsQ0FEVDtBQUVFLFdBQUssRUFBRTtBQUNMQyxlQUFPLEVBQUUsVUFESjtBQUVMQyxhQUFLLFlBQUtDLE1BQU0sR0FDWlAsbURBQUssQ0FBQ0MsTUFBTixDQUFhTyxZQUFiLENBQTBCQyxPQURkLEdBRVpULG1EQUFLLENBQUNDLE1BQU4sQ0FBYU8sWUFBYixDQUEwQkUsS0FGekI7QUFGQSxPQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FVR1gsU0FBUyxHQUFHLE1BQUMseUNBQUQ7QUFBTSxVQUFJLEVBQUMsT0FBWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQUgsR0FBMkJZLHlEQUFTLENBQUNyQixJQUFELEVBQU9RLElBQVAsQ0FWaEQsQ0FQRixDQURGO0FBc0JELEdBbENBO0FBQUEsWUFPNkJKLDREQVA3QjtBQUFBLEtBREgsQ0FERixDQURGO0FBeUNELENBN0NNOztJQUFNZCxjO1VBQ2dCRSw0RTs7O0tBRGhCRixjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmI4NGVlMWVlNDVmYzIwMWM2OTQ5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IFNwaW4sIFR5cG9ncmFwaHkgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7XHJcbiAgQ3VzdG9tRm9ybSxcclxuICBDdXRvbUZvcm1JdGVtLFxyXG59IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XHJcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSB9IGZyb20gJy4uL3Bsb3RzL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcywgSW5mb1Byb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBtYWluX3J1bl9pbmZvIH0gZnJvbSAnLi4vY29uc3RhbnRzJztcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnO1xyXG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQgeyBnZXRfbGFiZWwgfSBmcm9tICcuLi91dGlscyc7XHJcbmNvbnN0IHsgVGl0bGUgfSA9IFR5cG9ncmFwaHk7XHJcblxyXG5pbnRlcmZhY2UgTGl2ZU1vZGVIZWFkZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBMaXZlTW9kZUhlYWRlciA9ICh7IHF1ZXJ5IH06IExpdmVNb2RlSGVhZGVyUHJvcHMpID0+IHtcclxuICBjb25zdCB7IG5vdF9vbGRlcl90aGFuIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xyXG4gIGNvbnN0IGdsb2JhbFN0YXRlID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8PlxyXG4gICAgICA8Q3VzdG9tRm9ybSBkaXNwbGF5PVwiZmxleFwiIHN0eWxlPXt7IGFsaWduSXRlbXM6ICdjZW50ZXInLCB9fT5cclxuICAgICAgICB7bWFpbl9ydW5faW5mby5tYXAoKGluZm86IEluZm9Qcm9wcykgPT4ge1xyXG4gICAgICAgICAgY29uc3QgcGFyYW1zX2Zvcl9hcGkgPSBGb3JtYXRQYXJhbXNGb3JBUEkoXHJcbiAgICAgICAgICAgIGdsb2JhbFN0YXRlLFxyXG4gICAgICAgICAgICBxdWVyeSxcclxuICAgICAgICAgICAgaW5mby52YWx1ZSxcclxuICAgICAgICAgICAgJ0hMVC9FdmVudEluZm8nXHJcbiAgICAgICAgICApO1xyXG4gICAgICAgICAgY29uc3QgeyBkYXRhLCBpc0xvYWRpbmcgfSA9IHVzZVJlcXVlc3QoXHJcbiAgICAgICAgICAgIGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSxcclxuICAgICAgICAgICAge30sXHJcbiAgICAgICAgICAgIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXIsIG5vdF9vbGRlcl90aGFuXVxyXG4gICAgICAgICAgKTtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxDdXRvbUZvcm1JdGVtXHJcbiAgICAgICAgICAgICAgc3BhY2U9XCI4XCJcclxuICAgICAgICAgICAgICB3aWR0aD1cImZpdC1jb250ZW50XCJcclxuICAgICAgICAgICAgICBjb2xvcj17dGhlbWUuY29sb3JzLmNvbW1vbi53aGl0ZX1cclxuICAgICAgICAgICAgICBuYW1lPXtpbmZvLmxhYmVsfVxyXG4gICAgICAgICAgICAgIGxhYmVsPXtpbmZvLmxhYmVsfVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPFRpdGxlXHJcbiAgICAgICAgICAgICAgICBsZXZlbD17NH1cclxuICAgICAgICAgICAgICAgIHN0eWxlPXt7XHJcbiAgICAgICAgICAgICAgICAgIGRpc3BsYXk6ICdjb250ZW50cycsXHJcbiAgICAgICAgICAgICAgICAgIGNvbG9yOiBgJHt1cGRhdGVcclxuICAgICAgICAgICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xyXG4gICAgICAgICAgICAgICAgICAgIDogdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5lcnJvclxyXG4gICAgICAgICAgICAgICAgICAgIH1gLFxyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gPFNwaW4gc2l6ZT1cInNtYWxsXCIgLz4gOiBnZXRfbGFiZWwoaW5mbywgZGF0YSl9XHJcbiAgICAgICAgICAgICAgPC9UaXRsZT5cclxuICAgICAgICAgICAgPC9DdXRvbUZvcm1JdGVtPlxyXG4gICAgICAgICAgKTtcclxuICAgICAgICB9KX1cclxuICAgICAgPC9DdXN0b21Gb3JtPlxyXG4gICAgPC8+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
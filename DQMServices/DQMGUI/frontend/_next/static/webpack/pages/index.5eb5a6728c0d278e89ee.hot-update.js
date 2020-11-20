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
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/liveModeHeader.tsx",
    _s2 = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];












var Title = antd__WEBPACK_IMPORTED_MODULE_1__["Typography"].Title;
var LiveModeHeader = function LiveModeHeader(_ref) {
  _s2();

  var _s = $RefreshSig$();

  var query = _ref.query;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__["useUpdateLiveMode"])(),
      update = _useUpdateLiveMode.update,
      set_update = _useUpdateLiveMode.set_update,
      not_older_than = _useUpdateLiveMode.not_older_than;

  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]);
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomForm"], {
    display: "flex",
    style: {
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 7
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_8__["main_run_info"].map(_s(function (info) {
    _s();

    var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["FormatParamsForAPI"])(globalState, query, info.value, '/HLT/EventInfo');

    var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_10__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number]),
        data = _useRequest.data,
        isLoading = _useRequest.isLoading;

    return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CutomFormItem"], {
      space: "8",
      width: "fit-content",
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.common.white,
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
        color: "".concat(update ? _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.error)
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
        lineNumber: 64,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_11__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__["useRequest"]];
  }))), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomCol"], {
    justifycontent: "flex-end",
    display: "flex",
    alignitems: "center",
    texttransform: "uppercase",
    color: update ? _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.error,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 7
    }
  }, "Live Mode", __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
    space: "2",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Tooltip"], {
    title: "Updating mode is ".concat(update ? 'on' : 'off'),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "primary",
    shape: "circle",
    onClick: function onClick() {
      set_update(!update);
    },
    icon: update ? __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["PauseOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 90,
        columnNumber: 30
      }
    }) : __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["PlayCircleOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 90,
        columnNumber: 50
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
      columnNumber: 13
    }
  })))));
};

_s2(LiveModeHeader, "ohC5a37T9gYw9m5ORyIlJoPiizQ=", false, function () {
  return [_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__["useUpdateLiveMode"]];
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2xpdmVNb2RlSGVhZGVyLnRzeCJdLCJuYW1lcyI6WyJUaXRsZSIsIlR5cG9ncmFwaHkiLCJMaXZlTW9kZUhlYWRlciIsInF1ZXJ5IiwidXNlVXBkYXRlTGl2ZU1vZGUiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwibm90X29sZGVyX3RoYW4iLCJnbG9iYWxTdGF0ZSIsIlJlYWN0Iiwic3RvcmUiLCJhbGlnbkl0ZW1zIiwibWFpbl9ydW5faW5mbyIsIm1hcCIsImluZm8iLCJwYXJhbXNfZm9yX2FwaSIsIkZvcm1hdFBhcmFtc0ZvckFQSSIsInZhbHVlIiwidXNlUmVxdWVzdCIsImdldF9qcm9vdF9wbG90IiwiZGF0YXNldF9uYW1lIiwicnVuX251bWJlciIsImRhdGEiLCJpc0xvYWRpbmciLCJ0aGVtZSIsImNvbG9ycyIsImNvbW1vbiIsIndoaXRlIiwibGFiZWwiLCJkaXNwbGF5IiwiY29sb3IiLCJub3RpZmljYXRpb24iLCJzdWNjZXNzIiwiZXJyb3IiLCJnZXRfbGFiZWwiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUVBO0FBTUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtJQUNRQSxLLEdBQVVDLCtDLENBQVZELEs7QUFNRCxJQUFNRSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQW9DO0FBQUE7O0FBQUE7O0FBQUEsTUFBakNDLEtBQWlDLFFBQWpDQSxLQUFpQzs7QUFBQSwyQkFDakJDLG9GQUFpQixFQURBO0FBQUEsTUFDeERDLE1BRHdELHNCQUN4REEsTUFEd0Q7QUFBQSxNQUNoREMsVUFEZ0Qsc0JBQ2hEQSxVQURnRDtBQUFBLE1BQ3BDQyxjQURvQyxzQkFDcENBLGNBRG9DOztBQUVoRSxNQUFNQyxXQUFXLEdBQUdDLGdEQUFBLENBQWlCQywrREFBakIsQ0FBcEI7QUFDQSxTQUNFLDREQUNFLE1BQUMsNERBQUQ7QUFBWSxXQUFPLEVBQUMsTUFBcEI7QUFBMkIsU0FBSyxFQUFFO0FBQUVDLGdCQUFVLEVBQUU7QUFBZCxLQUFsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dDLHdEQUFhLENBQUNDLEdBQWQsSUFBa0IsVUFBQ0MsSUFBRCxFQUFxQjtBQUFBOztBQUN0QyxRQUFNQyxjQUFjLEdBQUdDLHVGQUFrQixDQUN2Q1IsV0FEdUMsRUFFdkNMLEtBRnVDLEVBR3ZDVyxJQUFJLENBQUNHLEtBSGtDLEVBSXZDLGdCQUp1QyxDQUF6Qzs7QUFEc0Msc0JBUVZDLG9FQUFVLENBQ3BDQyxzRUFBYyxDQUFDSixjQUFELENBRHNCLEVBRXBDLEVBRm9DLEVBR3BDLENBQUNaLEtBQUssQ0FBQ2lCLFlBQVAsRUFBcUJqQixLQUFLLENBQUNrQixVQUEzQixDQUhvQyxDQVJBO0FBQUEsUUFROUJDLElBUjhCLGVBUTlCQSxJQVI4QjtBQUFBLFFBUXhCQyxTQVJ3QixlQVF4QkEsU0FSd0I7O0FBYXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUVDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsTUFBYixDQUFvQkMsS0FIN0I7QUFJRSxVQUFJLEVBQUViLElBQUksQ0FBQ2MsS0FKYjtBQUtFLFdBQUssRUFBRWQsSUFBSSxDQUFDYyxLQUxkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FPRSxNQUFDLEtBQUQ7QUFDRSxXQUFLLEVBQUUsQ0FEVDtBQUVFLFdBQUssRUFBRTtBQUNMQyxlQUFPLEVBQUUsVUFESjtBQUVMQyxhQUFLLFlBQ0h6QixNQUFNLEdBQ0ZtQixtREFBSyxDQUFDQyxNQUFOLENBQWFNLFlBQWIsQ0FBMEJDLE9BRHhCLEdBRUZSLG1EQUFLLENBQUNDLE1BQU4sQ0FBYU0sWUFBYixDQUEwQkUsS0FIM0I7QUFGQSxPQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FXR1YsU0FBUyxHQUFHLE1BQUMseUNBQUQ7QUFBTSxVQUFJLEVBQUMsT0FBWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQUgsR0FBMkJXLHlEQUFTLENBQUNwQixJQUFELEVBQU9RLElBQVAsQ0FYaEQsQ0FQRixDQURGO0FBdUJELEdBcENBO0FBQUEsWUFRNkJKLDREQVI3QjtBQUFBLEtBREgsQ0FERixFQXdDRSxNQUFDLDJEQUFEO0FBQ0Usa0JBQWMsRUFBQyxVQURqQjtBQUVFLFdBQU8sRUFBQyxNQUZWO0FBR0UsY0FBVSxFQUFDLFFBSGI7QUFJRSxpQkFBYSxFQUFDLFdBSmhCO0FBS0UsU0FBSyxFQUNIYixNQUFNLEdBQ0ZtQixtREFBSyxDQUFDQyxNQUFOLENBQWFNLFlBQWIsQ0FBMEJDLE9BRHhCLEdBRUZSLG1EQUFLLENBQUNDLE1BQU4sQ0FBYU0sWUFBYixDQUEwQkUsS0FSbEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxrQkFZRSxNQUFDLDJEQUFEO0FBQVcsU0FBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDRDQUFEO0FBQVMsU0FBSyw2QkFBc0I1QixNQUFNLEdBQUcsSUFBSCxHQUFVLEtBQXRDLENBQWQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFDRSxRQUFJLEVBQUMsU0FEUDtBQUVFLFNBQUssRUFBQyxRQUZSO0FBR0UsV0FBTyxFQUFFLG1CQUFNO0FBQ2JDLGdCQUFVLENBQUMsQ0FBQ0QsTUFBRixDQUFWO0FBQ0QsS0FMSDtBQU1FLFFBQUksRUFBRUEsTUFBTSxHQUFHLE1BQUMsK0RBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUFILEdBQXVCLE1BQUMsb0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQU5yQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FERixDQVpGLENBeENGLENBREY7QUFvRUQsQ0F2RU07O0lBQU1ILGM7VUFDb0NFLDRFOzs7S0FEcENGLGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNWViNWE2NzI4YzBkMjc4ZTg5ZWUuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IEJ1dHRvbiwgVG9vbHRpcCwgU3BpbiwgVHlwb2dyYXBoeSB9IGZyb20gJ2FudGQnO1xuaW1wb3J0IHsgUGF1c2VPdXRsaW5lZCwgUGxheUNpcmNsZU91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xuXG5pbXBvcnQge1xuICBDdXN0b21Db2wsXG4gIEN1c3RvbURpdixcbiAgQ3VzdG9tRm9ybSxcbiAgQ3V0b21Gb3JtSXRlbSxcbn0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XG5pbXBvcnQgeyB1c2VVcGRhdGVMaXZlTW9kZSB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVVwZGF0ZUluTGl2ZU1vZGUnO1xuaW1wb3J0IHsgRm9ybWF0UGFyYW1zRm9yQVBJIH0gZnJvbSAnLi4vcGxvdHMvcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7IFF1ZXJ5UHJvcHMsIEluZm9Qcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IG1haW5fcnVuX2luZm8gfSBmcm9tICcuLi9jb25zdGFudHMnO1xuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnO1xuaW1wb3J0IHsgZ2V0X2pyb290X3Bsb3QgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IGdldF9sYWJlbCB9IGZyb20gJy4uL3V0aWxzJztcbmNvbnN0IHsgVGl0bGUgfSA9IFR5cG9ncmFwaHk7XG5cbmludGVyZmFjZSBMaXZlTW9kZUhlYWRlclByb3BzIHtcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XG59XG5cbmV4cG9ydCBjb25zdCBMaXZlTW9kZUhlYWRlciA9ICh7IHF1ZXJ5IH06IExpdmVNb2RlSGVhZGVyUHJvcHMpID0+IHtcbiAgY29uc3QgeyB1cGRhdGUsIHNldF91cGRhdGUsIG5vdF9vbGRlcl90aGFuIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xuICBjb25zdCBnbG9iYWxTdGF0ZSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xuICByZXR1cm4gKFxuICAgIDw+XG4gICAgICA8Q3VzdG9tRm9ybSBkaXNwbGF5PVwiZmxleFwiIHN0eWxlPXt7IGFsaWduSXRlbXM6ICdjZW50ZXInLH19PlxuICAgICAgICB7bWFpbl9ydW5faW5mby5tYXAoKGluZm86IEluZm9Qcm9wcykgPT4ge1xuICAgICAgICAgIGNvbnN0IHBhcmFtc19mb3JfYXBpID0gRm9ybWF0UGFyYW1zRm9yQVBJKFxuICAgICAgICAgICAgZ2xvYmFsU3RhdGUsXG4gICAgICAgICAgICBxdWVyeSxcbiAgICAgICAgICAgIGluZm8udmFsdWUsXG4gICAgICAgICAgICAnL0hMVC9FdmVudEluZm8nXG4gICAgICAgICAgKTtcblxuICAgICAgICAgIGNvbnN0IHsgZGF0YSwgaXNMb2FkaW5nIH0gPSB1c2VSZXF1ZXN0KFxuICAgICAgICAgICAgZ2V0X2pyb290X3Bsb3QocGFyYW1zX2Zvcl9hcGkpLFxuICAgICAgICAgICAge30sXG4gICAgICAgICAgICBbcXVlcnkuZGF0YXNldF9uYW1lLCBxdWVyeS5ydW5fbnVtYmVyLCBdXG4gICAgICAgICAgKTtcbiAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgPEN1dG9tRm9ybUl0ZW1cbiAgICAgICAgICAgICAgc3BhY2U9XCI4XCJcbiAgICAgICAgICAgICAgd2lkdGg9XCJmaXQtY29udGVudFwiXG4gICAgICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuY29tbW9uLndoaXRlfVxuICAgICAgICAgICAgICBuYW1lPXtpbmZvLmxhYmVsfVxuICAgICAgICAgICAgICBsYWJlbD17aW5mby5sYWJlbH1cbiAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgPFRpdGxlXG4gICAgICAgICAgICAgICAgbGV2ZWw9ezR9XG4gICAgICAgICAgICAgICAgc3R5bGU9e3tcbiAgICAgICAgICAgICAgICAgIGRpc3BsYXk6ICdjb250ZW50cycsXG4gICAgICAgICAgICAgICAgICBjb2xvcjogYCR7XG4gICAgICAgICAgICAgICAgICAgIHVwZGF0ZVxuICAgICAgICAgICAgICAgICAgICAgID8gdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5zdWNjZXNzXG4gICAgICAgICAgICAgICAgICAgICAgOiB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLmVycm9yXG4gICAgICAgICAgICAgICAgICB9YCxcbiAgICAgICAgICAgICAgICB9fVxuICAgICAgICAgICAgICA+XG4gICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IDxTcGluIHNpemU9XCJzbWFsbFwiIC8+IDogZ2V0X2xhYmVsKGluZm8sIGRhdGEpfVxuICAgICAgICAgICAgICA8L1RpdGxlPlxuICAgICAgICAgICAgPC9DdXRvbUZvcm1JdGVtPlxuICAgICAgICAgICk7XG4gICAgICAgIH0pfVxuICAgICAgPC9DdXN0b21Gb3JtPlxuICAgICAgPEN1c3RvbUNvbFxuICAgICAgICBqdXN0aWZ5Y29udGVudD1cImZsZXgtZW5kXCJcbiAgICAgICAgZGlzcGxheT1cImZsZXhcIlxuICAgICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcbiAgICAgICAgdGV4dHRyYW5zZm9ybT1cInVwcGVyY2FzZVwiXG4gICAgICAgIGNvbG9yPXtcbiAgICAgICAgICB1cGRhdGVcbiAgICAgICAgICAgID8gdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5zdWNjZXNzXG4gICAgICAgICAgICA6IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uZXJyb3JcbiAgICAgICAgfVxuICAgICAgPlxuICAgICAgICBMaXZlIE1vZGVcbiAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj5cbiAgICAgICAgICA8VG9vbHRpcCB0aXRsZT17YFVwZGF0aW5nIG1vZGUgaXMgJHt1cGRhdGUgPyAnb24nIDogJ29mZid9YH0+XG4gICAgICAgICAgICA8QnV0dG9uXG4gICAgICAgICAgICAgIHR5cGU9XCJwcmltYXJ5XCJcbiAgICAgICAgICAgICAgc2hhcGU9XCJjaXJjbGVcIlxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XG4gICAgICAgICAgICAgICAgc2V0X3VwZGF0ZSghdXBkYXRlKTtcbiAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgaWNvbj17dXBkYXRlID8gPFBhdXNlT3V0bGluZWQgLz4gOiA8UGxheUNpcmNsZU91dGxpbmVkIC8+fVxuICAgICAgICAgICAgPjwvQnV0dG9uPlxuICAgICAgICAgIDwvVG9vbHRpcD5cbiAgICAgICAgPC9DdXN0b21EaXY+XG4gICAgICA8L0N1c3RvbUNvbD5cbiAgICA8Lz5cbiAgKTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9
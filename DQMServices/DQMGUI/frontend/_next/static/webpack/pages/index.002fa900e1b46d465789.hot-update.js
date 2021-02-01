webpackHotUpdate_N_E("pages/index",{

/***/ "./components/browsing/runsBrowser.tsx":
/*!*********************************************!*\
  !*** ./components/browsing/runsBrowser.tsx ***!
  \*********************************************/
/*! exports provided: RunBrowser */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "RunBrowser", function() { return RunBrowser; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _hooks_useSearch__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../hooks/useSearch */ "./hooks/useSearch.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/browsing/runsBrowser.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;






var Option = antd__WEBPACK_IMPORTED_MODULE_1__["Select"].Option;
var RunBrowser = function RunBrowser(_ref) {
  _s();

  var query = _ref.query,
      setCurrentRunNumber = _ref.setCurrentRunNumber,
      withoutArrows = _ref.withoutArrows,
      withoutLabel = _ref.withoutLabel,
      current_run_number = _ref.current_run_number,
      current_dataset_name = _ref.current_dataset_name;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(false),
      openSelect = _useState[0],
      setSelect = _useState[1];

  var styledSelectRef = Object(react__WEBPACK_IMPORTED_MODULE_0__["useRef"])(null);

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      styledSelectWidth = _useState2[0],
      setStyledSelect = _useState2[1];

  Object(react__WEBPACK_IMPORTED_MODULE_0__["useEffect"])(function () {
    if (styledSelectRef.current && styledSelectRef.current.clientWidth) {
      setStyledSelect(styledSelectRef.current.clientWidth);
    }
  }, []);

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      currentRunNumberIndex = _useState3[0],
      setCurrentRunNumberIndex = _useState3[1];

  var dataset_name = current_dataset_name ? current_dataset_name : query.dataset_name;

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_5__["useSearch"])('', dataset_name),
      results_grouped = _useSearch.results_grouped,
      isLoading = _useSearch.isLoading;

  var runNumbers = results_grouped[0] ? results_grouped[0].runs.map(function (run) {
    return run.toString();
  }) : [];
  Object(react__WEBPACK_IMPORTED_MODULE_0__["useEffect"])(function () {
    var query_run_number = current_run_number ? current_run_number.toString() : query.run_number;
    setCurrentRunNumberIndex(runNumbers.indexOf(query_run_number));
  }, [runNumbers, isLoading]);
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 62,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledFormItem"], {
    labelcolor: "white",
    name: 'dataset_name',
    label: "".concat(!withoutLabel ? 'Run' : ''),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 63,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 9
    }
  }, !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    disabled: !runNumbers[currentRunNumberIndex - 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 73,
        columnNumber: 23
      }
    }),
    type: "link",
    onClick: function onClick() {
      setCurrentRunNumber(runNumbers[currentRunNumberIndex - 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 71,
      columnNumber: 15
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 11
    }
  }, __jsx("div", {
    ref: styledSelectRef,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 13
    }
  }, __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSelect"], {
    width: "".concat(styledSelectWidth.toString(), "px"),
    onClick: function onClick() {
      return setSelect(!openSelect);
    },
    value: runNumbers[currentRunNumberIndex],
    onChange: function onChange(e) {
      setCurrentRunNumber(e);
      setSelect(!openSelect);
    },
    showSearch: true,
    open: openSelect,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 15
    }
  }, runNumbers && runNumbers.map(function (run) {
    return __jsx(Option, {
      onClick: function onClick() {
        setSelect(false);
      },
      value: run,
      key: run.toString(),
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 97,
        columnNumber: 23
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 105,
        columnNumber: 27
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 106,
        columnNumber: 29
      }
    })) : __jsx("div", {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 109,
        columnNumber: 29
      }
    }, run));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 120,
        columnNumber: 23
      }
    }),
    disabled: !runNumbers[currentRunNumberIndex + 1],
    type: "link",
    onClick: function onClick() {
      setCurrentRunNumber(runNumbers[currentRunNumberIndex + 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 15
    }
  })))));
};

_s(RunBrowser, "bmEMcnhpd9JXdO+4ZZYM9mREKQo=", false, function () {
  return [_hooks_useSearch__WEBPACK_IMPORTED_MODULE_5__["useSearch"]];
});

_c = RunBrowser;

var _c;

$RefreshReg$(_c, "RunBrowser");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInN0eWxlZFNlbGVjdFJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFdpZHRoIiwic2V0U3R5bGVkU2VsZWN0IiwidXNlRWZmZWN0IiwiY3VycmVudCIsImNsaWVudFdpZHRoIiwiY3VycmVudFJ1bk51bWJlckluZGV4Iiwic2V0Q3VycmVudFJ1bk51bWJlckluZGV4IiwiZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwiaXNMb2FkaW5nIiwicnVuTnVtYmVycyIsInJ1bnMiLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsInF1ZXJ5X3J1bl9udW1iZXIiLCJydW5fbnVtYmVyIiwiaW5kZXhPZiIsImUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFJQTtJQUdRQSxNLEdBQVdDLDJDLENBQVhELE07QUFZRCxJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQU9IO0FBQUE7O0FBQUEsTUFOckJDLEtBTXFCLFFBTnJCQSxLQU1xQjtBQUFBLE1BTHJCQyxtQkFLcUIsUUFMckJBLG1CQUtxQjtBQUFBLE1BSnJCQyxhQUlxQixRQUpyQkEsYUFJcUI7QUFBQSxNQUhyQkMsWUFHcUIsUUFIckJBLFlBR3FCO0FBQUEsTUFGckJDLGtCQUVxQixRQUZyQkEsa0JBRXFCO0FBQUEsTUFEckJDLG9CQUNxQixRQURyQkEsb0JBQ3FCOztBQUFBLGtCQUNXQyxz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TUFEUjtBQUVFLFlBQVEsRUFBRSxDQUFDb0IsVUFBVSxDQUFDTixxQkFBcUIsR0FBRyxDQUF6QixDQUZ2QjtBQUdFLFFBQUksRUFBQyxNQUhQO0FBSUUsV0FBTyxFQUFFLG1CQUFNO0FBQ2JmLHlCQUFtQixDQUFDcUIsVUFBVSxDQUFDTixxQkFBcUIsR0FBRyxDQUF6QixDQUFYLENBQW5CO0FBQ0QsS0FOSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FsREosQ0FMRixDQURGLENBREY7QUF3RUQsQ0E1R007O0dBQU1qQixVO1VBdUI0Qm9CLDBEOzs7S0F2QjVCcEIsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4wMDJmYTkwMGUxYjQ2ZDQ2NTc4OS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QsIHVzZVJlZiB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgQ29sLCBSb3csIFNlbGVjdCwgU3BpbiwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IENhcmV0UmlnaHRGaWxsZWQsIENhcmV0TGVmdEZpbGxlZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkU2VsZWN0LFxyXG4gIE9wdGlvblBhcmFncmFwaCxcclxufSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcblxyXG5jb25zdCB7IE9wdGlvbiB9ID0gU2VsZWN0O1xyXG5cclxuaW50ZXJmYWNlIFJ1bkJyb3dzZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcihjdXJyZW50UnVuTnVtYmVyOiBzdHJpbmcpOiB2b2lkO1xyXG4gIHdpdGhvdXRBcnJvd3M/OiBib29sZWFuO1xyXG4gIHdpdGhvdXRMYWJlbD86IGJvb2xlYW47XHJcbiAgc2VsZWN0b3JXaWR0aD86IHN0cmluZztcclxuICBjdXJyZW50X3J1bl9udW1iZXI/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBSdW5Ccm93c2VyID0gKHtcclxuICBxdWVyeSxcclxuICBzZXRDdXJyZW50UnVuTnVtYmVyLFxyXG4gIHdpdGhvdXRBcnJvd3MsXHJcbiAgd2l0aG91dExhYmVsLFxyXG4gIGN1cnJlbnRfcnVuX251bWJlcixcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZSxcclxufTogUnVuQnJvd3NlclByb3BzKSA9PiB7XHJcbiAgY29uc3QgW29wZW5TZWxlY3QsIHNldFNlbGVjdF0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3Qgc3R5bGVkU2VsZWN0UmVmID0gdXNlUmVmKG51bGwpXHJcblxyXG4gIGNvbnN0IFtzdHlsZWRTZWxlY3RXaWR0aCwgc2V0U3R5bGVkU2VsZWN0XSA9IHVzZVN0YXRlKDApXHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBpZiAoc3R5bGVkU2VsZWN0UmVmLmN1cnJlbnQgJiYgc3R5bGVkU2VsZWN0UmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpIHtcclxuICAgICAgc2V0U3R5bGVkU2VsZWN0KHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKVxyXG4gICAgfVxyXG4gIH0sIFtdKVxyXG5cclxuICBjb25zdCBbY3VycmVudFJ1bk51bWJlckluZGV4LCBzZXRDdXJyZW50UnVuTnVtYmVySW5kZXhdID0gdXNlU3RhdGU8bnVtYmVyPigwKTtcclxuICBjb25zdCBkYXRhc2V0X25hbWUgPSBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgPyBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgOiBxdWVyeS5kYXRhc2V0X25hbWU7XHJcbiAgY29uc3QgeyByZXN1bHRzX2dyb3VwZWQsIGlzTG9hZGluZyB9ID0gdXNlU2VhcmNoKCcnLCBkYXRhc2V0X25hbWUpO1xyXG5cclxuICBjb25zdCBydW5OdW1iZXJzID0gcmVzdWx0c19ncm91cGVkWzBdXHJcbiAgICA/IHJlc3VsdHNfZ3JvdXBlZFswXS5ydW5zLm1hcCgocnVuOiBudW1iZXIpID0+IHJ1bi50b1N0cmluZygpKVxyXG4gICAgOiBbXTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHF1ZXJ5X3J1bl9udW1iZXIgPSBjdXJyZW50X3J1bl9udW1iZXJcclxuICAgICAgPyBjdXJyZW50X3J1bl9udW1iZXIudG9TdHJpbmcoKVxyXG4gICAgICA6IHF1ZXJ5LnJ1bl9udW1iZXI7XHJcbiAgICBzZXRDdXJyZW50UnVuTnVtYmVySW5kZXgocnVuTnVtYmVycy5pbmRleE9mKHF1ZXJ5X3J1bl9udW1iZXIpKTtcclxuICB9LCBbcnVuTnVtYmVycywgaXNMb2FkaW5nXSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Q29sPlxyXG4gICAgICA8U3R5bGVkRm9ybUl0ZW1cclxuICAgICAgICBsYWJlbGNvbG9yPVwid2hpdGVcIlxyXG4gICAgICAgIG5hbWU9eydkYXRhc2V0X25hbWUnfVxyXG4gICAgICAgIGxhYmVsPXtgJHshd2l0aG91dExhYmVsID8gJ1J1bicgOiAnJ31gfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFJvdyBqdXN0aWZ5PVwiY2VudGVyXCIgYWxpZ249XCJtaWRkbGVcIj5cclxuICAgICAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCAtIDFdfVxyXG4gICAgICAgICAgICAgICAgaWNvbj17PENhcmV0TGVmdEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCAtIDFdKTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICApfVxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPGRpdiByZWY9e3N0eWxlZFNlbGVjdFJlZn0+XHJcbiAgICAgICAgICAgICAgPFN0eWxlZFNlbGVjdFxyXG4gICAgICAgICAgICAgICAgd2lkdGg9e2Ake3N0eWxlZFNlbGVjdFdpZHRoLnRvU3RyaW5nKCl9cHhgfVxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleF19XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKGUpO1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtydW5OdW1iZXJzICYmXHJcbiAgICAgICAgICAgICAgICAgIHJ1bk51bWJlcnMubWFwKChydW46IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZT17cnVufVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e3J1bi50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGRpdj57cnVufTwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICl9XHJcbiAgICAgICAgICAgICAgICAgICAgICA8L09wdGlvbj5cclxuICAgICAgICAgICAgICAgICAgICApO1xyXG4gICAgICAgICAgICAgICAgICB9KX1cclxuICAgICAgICAgICAgICA8L1N0eWxlZFNlbGVjdD5cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICAgICAgaWNvbj17PENhcmV0UmlnaHRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICBkaXNhYmxlZD17IXJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV19XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggKyAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cclxuICAgIDwvQ29sPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
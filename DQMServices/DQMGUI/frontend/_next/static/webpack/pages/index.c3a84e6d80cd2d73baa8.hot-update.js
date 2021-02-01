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

  var runOptionRef = Object(react__WEBPACK_IMPORTED_MODULE_0__["useRef"])(null);
  var styledSelectRef = Object(react__WEBPACK_IMPORTED_MODULE_0__["useRef"])(null);

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      runOptionWidth = _useState2[0],
      setRunOptionWidth = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      styledSelectWidth = _useState3[0],
      setStyledSelect = _useState3[1];

  Object(react__WEBPACK_IMPORTED_MODULE_0__["useEffect"])(function () {
    if (runOptionRef.current && runOptionRef.current.clientWidth) {
      setStyledSelect(runOptionRef.current.clientWidth);
    }
  }, []);

  var _useState4 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      currentRunNumberIndex = _useState4[0],
      setCurrentRunNumberIndex = _useState4[1];

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
      lineNumber: 64,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledFormItem"], {
    labelcolor: "white",
    name: 'dataset_name',
    label: "".concat(!withoutLabel ? 'Run' : ''),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 65,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 9
    }
  }, !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    disabled: !runNumbers[currentRunNumberIndex - 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 75,
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
      lineNumber: 73,
      columnNumber: 15
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 11
    }
  }, __jsx("div", {
    ref: styledSelectRef,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
      columnNumber: 13
    }
  }, __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSelect"], {
    width: styledSelectWidth === 0 ? 'fit-content' : styledSelectWidth.toString(),
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
      lineNumber: 85,
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
        lineNumber: 99,
        columnNumber: 23
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 107,
        columnNumber: 27
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 108,
        columnNumber: 29
      }
    })) : __jsx("div", {
      ref: runOptionRef,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 111,
        columnNumber: 29
      }
    }, run));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 120,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 122,
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
      lineNumber: 121,
      columnNumber: 15
    }
  })))));
};

_s(RunBrowser, "UdT8ddRUyphG2E3hYV/PIgbia8M=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInJ1bk9wdGlvblJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFJlZiIsInJ1bk9wdGlvbldpZHRoIiwic2V0UnVuT3B0aW9uV2lkdGgiLCJzdHlsZWRTZWxlY3RXaWR0aCIsInNldFN0eWxlZFNlbGVjdCIsInVzZUVmZmVjdCIsImN1cnJlbnQiLCJjbGllbnRXaWR0aCIsImN1cnJlbnRSdW5OdW1iZXJJbmRleCIsInNldEN1cnJlbnRSdW5OdW1iZXJJbmRleCIsImRhdGFzZXRfbmFtZSIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsInJ1bk51bWJlcnMiLCJydW5zIiwibWFwIiwicnVuIiwidG9TdHJpbmciLCJxdWVyeV9ydW5fbnVtYmVyIiwicnVuX251bWJlciIsImluZGV4T2YiLCJlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBSUE7SUFHUUEsTSxHQUFXQywyQyxDQUFYRCxNO0FBWUQsSUFBTUUsVUFBVSxHQUFHLFNBQWJBLFVBQWEsT0FPSDtBQUFBOztBQUFBLE1BTnJCQyxLQU1xQixRQU5yQkEsS0FNcUI7QUFBQSxNQUxyQkMsbUJBS3FCLFFBTHJCQSxtQkFLcUI7QUFBQSxNQUpyQkMsYUFJcUIsUUFKckJBLGFBSXFCO0FBQUEsTUFIckJDLFlBR3FCLFFBSHJCQSxZQUdxQjtBQUFBLE1BRnJCQyxrQkFFcUIsUUFGckJBLGtC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O0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBeUJtQixHQUF6QixDQVpOLENBREY7QUFpQkQsR0FsQkQsQ0FaSixDQURGLENBREYsQ0FiRixFQWlERyxDQUFDMUIsYUFBRCxJQUNDLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFDRSxRQUFJLEVBQUUsTUFBQyxrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BRFI7QUFFRSxZQUFRLEVBQUUsQ0FBQ3VCLFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FGdkI7QUFHRSxRQUFJLEVBQUMsTUFIUDtBQUlFLFdBQU8sRUFBRSxtQkFBTTtBQUNibEIseUJBQW1CLENBQUN3QixVQUFVLENBQUNOLHFCQUFxQixHQUFHLENBQXpCLENBQVgsQ0FBbkI7QUFDRCxLQU5IO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQWxESixDQUxGLENBREYsQ0FERjtBQXdFRCxDQTlHTTs7R0FBTXBCLFU7VUF5QjRCdUIsMEQ7OztLQXpCNUJ2QixVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmMzYTg0ZTZkODBjZDJkNzNiYWE4LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCwgdXNlUmVmIH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBDb2wsIFJvdywgU2VsZWN0LCBTcGluLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgQ2FyZXRSaWdodEZpbGxlZCwgQ2FyZXRMZWZ0RmlsbGVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0gfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRTZWxlY3QsXHJcbiAgT3B0aW9uUGFyYWdyYXBoLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdXNlU2VhcmNoIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlU2VhcmNoJztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuXHJcbmNvbnN0IHsgT3B0aW9uIH0gPSBTZWxlY3Q7XHJcblxyXG5pbnRlcmZhY2UgUnVuQnJvd3NlclByb3BzIHtcclxuICBxdWVyeTogUXVlcnlQcm9wcztcclxuICBzZXRDdXJyZW50UnVuTnVtYmVyKGN1cnJlbnRSdW5OdW1iZXI6IHN0cmluZyk6IHZvaWQ7XHJcbiAgd2l0aG91dEFycm93cz86IGJvb2xlYW47XHJcbiAgd2l0aG91dExhYmVsPzogYm9vbGVhbjtcclxuICBzZWxlY3RvcldpZHRoPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfcnVuX251bWJlcj86IHN0cmluZztcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZT86IHN0cmluZztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFJ1bkJyb3dzZXIgPSAoe1xyXG4gIHF1ZXJ5LFxyXG4gIHNldEN1cnJlbnRSdW5OdW1iZXIsXHJcbiAgd2l0aG91dEFycm93cyxcclxuICB3aXRob3V0TGFiZWwsXHJcbiAgY3VycmVudF9ydW5fbnVtYmVyLFxyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lLFxyXG59OiBSdW5Ccm93c2VyUHJvcHMpID0+IHtcclxuICBjb25zdCBbb3BlblNlbGVjdCwgc2V0U2VsZWN0XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBydW5PcHRpb25SZWYgPSB1c2VSZWYobnVsbClcclxuICBjb25zdCBzdHlsZWRTZWxlY3RSZWYgPSB1c2VSZWYobnVsbClcclxuXHJcbiAgY29uc3QgW3J1bk9wdGlvbldpZHRoLCBzZXRSdW5PcHRpb25XaWR0aF0gPSB1c2VTdGF0ZSgwKVxyXG4gIGNvbnN0IFtzdHlsZWRTZWxlY3RXaWR0aCwgc2V0U3R5bGVkU2VsZWN0XSA9IHVzZVN0YXRlKDApXHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBpZiAocnVuT3B0aW9uUmVmLmN1cnJlbnQgJiYgcnVuT3B0aW9uUmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpIHtcclxuICAgICAgc2V0U3R5bGVkU2VsZWN0KHJ1bk9wdGlvblJlZi5jdXJyZW50LmNsaWVudFdpZHRoKVxyXG4gICAgfVxyXG4gIH0sIFtdKVxyXG5cclxuICBjb25zdCBbY3VycmVudFJ1bk51bWJlckluZGV4LCBzZXRDdXJyZW50UnVuTnVtYmVySW5kZXhdID0gdXNlU3RhdGU8bnVtYmVyPigwKTtcclxuICBjb25zdCBkYXRhc2V0X25hbWUgPSBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgPyBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgOiBxdWVyeS5kYXRhc2V0X25hbWU7XHJcbiAgY29uc3QgeyByZXN1bHRzX2dyb3VwZWQsIGlzTG9hZGluZyB9ID0gdXNlU2VhcmNoKCcnLCBkYXRhc2V0X25hbWUpO1xyXG5cclxuICBjb25zdCBydW5OdW1iZXJzID0gcmVzdWx0c19ncm91cGVkWzBdXHJcbiAgICA/IHJlc3VsdHNfZ3JvdXBlZFswXS5ydW5zLm1hcCgocnVuOiBudW1iZXIpID0+IHJ1bi50b1N0cmluZygpKVxyXG4gICAgOiBbXTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHF1ZXJ5X3J1bl9udW1iZXIgPSBjdXJyZW50X3J1bl9udW1iZXJcclxuICAgICAgPyBjdXJyZW50X3J1bl9udW1iZXIudG9TdHJpbmcoKVxyXG4gICAgICA6IHF1ZXJ5LnJ1bl9udW1iZXI7XHJcbiAgICBzZXRDdXJyZW50UnVuTnVtYmVySW5kZXgocnVuTnVtYmVycy5pbmRleE9mKHF1ZXJ5X3J1bl9udW1iZXIpKTtcclxuICB9LCBbcnVuTnVtYmVycywgaXNMb2FkaW5nXSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Q29sPlxyXG4gICAgICA8U3R5bGVkRm9ybUl0ZW1cclxuICAgICAgICBsYWJlbGNvbG9yPVwid2hpdGVcIlxyXG4gICAgICAgIG5hbWU9eydkYXRhc2V0X25hbWUnfVxyXG4gICAgICAgIGxhYmVsPXtgJHshd2l0aG91dExhYmVsID8gJ1J1bicgOiAnJ31gfVxyXG4gICAgICA+XHJcbiAgICAgICAgPFJvdyBqdXN0aWZ5PVwiY2VudGVyXCIgYWxpZ249XCJtaWRkbGVcIj5cclxuICAgICAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCAtIDFdfVxyXG4gICAgICAgICAgICAgICAgaWNvbj17PENhcmV0TGVmdEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCAtIDFdKTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICApfVxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPGRpdiByZWY9e3N0eWxlZFNlbGVjdFJlZn0+XHJcbiAgICAgICAgICAgICAgPFN0eWxlZFNlbGVjdFxyXG4gICAgICAgICAgICAgICAgd2lkdGg9e3N0eWxlZFNlbGVjdFdpZHRoID09PSAwID8gJ2ZpdC1jb250ZW50Jzogc3R5bGVkU2VsZWN0V2lkdGgudG9TdHJpbmcoKX1cclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHNldFNlbGVjdCghb3BlblNlbGVjdCl9XHJcbiAgICAgICAgICAgICAgICB2YWx1ZT17cnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXhdfVxyXG4gICAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBhbnkpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihlKTtcclxuICAgICAgICAgICAgICAgICAgc2V0U2VsZWN0KCFvcGVuU2VsZWN0KTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgICBzaG93U2VhcmNoPXt0cnVlfVxyXG4gICAgICAgICAgICAgICAgb3Blbj17b3BlblNlbGVjdH1cclxuICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICB7cnVuTnVtYmVycyAmJlxyXG4gICAgICAgICAgICAgICAgICBydW5OdW1iZXJzLm1hcCgocnVuOiBhbnkpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgPE9wdGlvblxyXG4gICAgICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgc2V0U2VsZWN0KGZhbHNlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgICAgICAgICAgdmFsdWU9e3J1bn1cclxuICAgICAgICAgICAgICAgICAgICAgICAga2V5PXtydW4udG9TdHJpbmcoKX1cclxuICAgICAgICAgICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IChcclxuICAgICAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPFNwaW4gLz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICA8L09wdGlvblBhcmFncmFwaD5cclxuICAgICAgICAgICAgICAgICAgICAgICAgKSA6IChcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxkaXYgcmVmPXtydW5PcHRpb25SZWZ9PntydW59PC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgKX1cclxuICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uPlxyXG4gICAgICAgICAgICAgICAgICAgICk7XHJcbiAgICAgICAgICAgICAgICAgIH0pfVxyXG4gICAgICAgICAgICAgIDwvU3R5bGVkU2VsZWN0PlxyXG4gICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgeyF3aXRob3V0QXJyb3dzICYmIChcclxuICAgICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgICBpY29uPXs8Q2FyZXRSaWdodEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggKyAxXX1cclxuICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdKTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICApfVxyXG4gICAgICAgIDwvUm93PlxyXG4gICAgICA8L1N0eWxlZEZvcm1JdGVtPlxyXG4gICAgPC9Db2w+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
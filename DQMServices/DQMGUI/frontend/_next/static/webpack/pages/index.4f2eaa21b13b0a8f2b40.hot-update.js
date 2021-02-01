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
    //@ts-ignore
    if (styledSelectRef.current && styledSelectRef.current.clientWidth) {
      //@ts-ignore
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
    width: '100px',
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInN0eWxlZFNlbGVjdFJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFdpZHRoIiwic2V0U3R5bGVkU2VsZWN0IiwidXNlRWZmZWN0IiwiY3VycmVudCIsImNsaWVudFdpZHRoIiwiY3VycmVudFJ1bk51bWJlckluZGV4Iiwic2V0Q3VycmVudFJ1bk51bWJlckluZGV4IiwiZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwiaXNMb2FkaW5nIiwicnVuTnVtYmVycyIsInJ1bnMiLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsInF1ZXJ5X3J1bl9udW1iZXIiLCJydW5fbnVtYmVyIiwiaW5kZXhPZiIsImUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFJQTtJQUdRQSxNLEdBQVdDLDJDLENBQVhELE07QUFZRCxJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQU9IO0FBQUE7O0FBQUEsTUFOckJDLEtBTXFCLFFBTnJCQSxLQU1xQjtBQUFBLE1BTHJCQyxtQkFLcUIsUUFMckJBLG1CQUtxQjtBQUFBLE1BSnJCQyxhQUlxQixRQUpyQkEsYUFJcUI7QUFBQSxNQUhyQkMsWUFHcUIsUUFIckJBLFlBR3FCO0FBQUEsTUFGckJDLGtCQUVxQixRQUZyQkEsa0JBRXFCO0FBQUEsTUFEckJDLG9CQUNxQixRQURyQkEsb0JBQ3FCOztBQUFBLGtCQUNXQyxz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QUFFLENBQUNvQixVQUFVLENBQUNOLHFCQUFxQixHQUFHLENBQXpCLENBRnZCO0FBR0UsUUFBSSxFQUFDLE1BSFA7QUFJRSxXQUFPLEVBQUUsbUJBQU07QUFDYmYseUJBQW1CLENBQUNxQixVQUFVLENBQUNOLHFCQUFxQixHQUFHLENBQXpCLENBQVgsQ0FBbkI7QUFDRCxLQU5IO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQWxESixDQUxGLENBREYsQ0FERjtBQXdFRCxDQTlHTTs7R0FBTWpCLFU7VUF5QjRCb0IsMEQ7OztLQXpCNUJwQixVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjRmMmVhYTIxYjEzYjBhOGYyYjQwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCwgdXNlUmVmIH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBDb2wsIFJvdywgU2VsZWN0LCBTcGluLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgQ2FyZXRSaWdodEZpbGxlZCwgQ2FyZXRMZWZ0RmlsbGVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0gfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRTZWxlY3QsXHJcbiAgT3B0aW9uUGFyYWdyYXBoLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdXNlU2VhcmNoIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlU2VhcmNoJztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuXHJcbmNvbnN0IHsgT3B0aW9uIH0gPSBTZWxlY3Q7XHJcblxyXG5pbnRlcmZhY2UgUnVuQnJvd3NlclByb3BzIHtcclxuICBxdWVyeTogUXVlcnlQcm9wcztcclxuICBzZXRDdXJyZW50UnVuTnVtYmVyKGN1cnJlbnRSdW5OdW1iZXI6IHN0cmluZyk6IHZvaWQ7XHJcbiAgd2l0aG91dEFycm93cz86IGJvb2xlYW47XHJcbiAgd2l0aG91dExhYmVsPzogYm9vbGVhbjtcclxuICBzZWxlY3RvcldpZHRoPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfcnVuX251bWJlcj86IHN0cmluZztcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZT86IHN0cmluZztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFJ1bkJyb3dzZXIgPSAoe1xyXG4gIHF1ZXJ5LFxyXG4gIHNldEN1cnJlbnRSdW5OdW1iZXIsXHJcbiAgd2l0aG91dEFycm93cyxcclxuICB3aXRob3V0TGFiZWwsXHJcbiAgY3VycmVudF9ydW5fbnVtYmVyLFxyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lLFxyXG59OiBSdW5Ccm93c2VyUHJvcHMpID0+IHtcclxuICBjb25zdCBbb3BlblNlbGVjdCwgc2V0U2VsZWN0XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBzdHlsZWRTZWxlY3RSZWYgPSB1c2VSZWYobnVsbClcclxuXHJcbiAgY29uc3QgW3N0eWxlZFNlbGVjdFdpZHRoLCBzZXRTdHlsZWRTZWxlY3RdID0gdXNlU3RhdGUoMClcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIC8vQHRzLWlnbm9yZVxyXG4gICAgaWYgKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50ICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIC8vQHRzLWlnbm9yZVxyXG4gICAgICBzZXRTdHlsZWRTZWxlY3Qoc3R5bGVkU2VsZWN0UmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpXHJcbiAgICB9XHJcbiAgfSwgW10pXHJcblxyXG4gIGNvbnN0IFtjdXJyZW50UnVuTnVtYmVySW5kZXgsIHNldEN1cnJlbnRSdW5OdW1iZXJJbmRleF0gPSB1c2VTdGF0ZTxudW1iZXI+KDApO1xyXG4gIGNvbnN0IGRhdGFzZXRfbmFtZSA9IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICA/IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICA6IHF1ZXJ5LmRhdGFzZXRfbmFtZTtcclxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgaXNMb2FkaW5nIH0gPSB1c2VTZWFyY2goJycsIGRhdGFzZXRfbmFtZSk7XHJcblxyXG4gIGNvbnN0IHJ1bk51bWJlcnMgPSByZXN1bHRzX2dyb3VwZWRbMF1cclxuICAgID8gcmVzdWx0c19ncm91cGVkWzBdLnJ1bnMubWFwKChydW46IG51bWJlcikgPT4gcnVuLnRvU3RyaW5nKCkpXHJcbiAgICA6IFtdO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgcXVlcnlfcnVuX251bWJlciA9IGN1cnJlbnRfcnVuX251bWJlclxyXG4gICAgICA/IGN1cnJlbnRfcnVuX251bWJlci50b1N0cmluZygpXHJcbiAgICAgIDogcXVlcnkucnVuX251bWJlcjtcclxuICAgIHNldEN1cnJlbnRSdW5OdW1iZXJJbmRleChydW5OdW1iZXJzLmluZGV4T2YocXVlcnlfcnVuX251bWJlcikpO1xyXG4gIH0sIFtydW5OdW1iZXJzLCBpc0xvYWRpbmddKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxDb2w+XHJcbiAgICAgIDxTdHlsZWRGb3JtSXRlbVxyXG4gICAgICAgIGxhYmVsY29sb3I9XCJ3aGl0ZVwiXHJcbiAgICAgICAgbmFtZT17J2RhdGFzZXRfbmFtZSd9XHJcbiAgICAgICAgbGFiZWw9e2AkeyF3aXRob3V0TGFiZWwgPyAnUnVuJyA6ICcnfWB9XHJcbiAgICAgID5cclxuICAgICAgICA8Um93IGp1c3RpZnk9XCJjZW50ZXJcIiBhbGlnbj1cIm1pZGRsZVwiPlxyXG4gICAgICAgICAgeyF3aXRob3V0QXJyb3dzICYmIChcclxuICAgICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgICBkaXNhYmxlZD17IXJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4IC0gMV19XHJcbiAgICAgICAgICAgICAgICBpY29uPXs8Q2FyZXRMZWZ0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4IC0gMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICA8ZGl2IHJlZj17c3R5bGVkU2VsZWN0UmVmfT5cclxuICAgICAgICAgICAgICA8U3R5bGVkU2VsZWN0XHJcbiAgICAgICAgICAgICAgICB3aWR0aD17JzEwMHB4J31cclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHNldFNlbGVjdCghb3BlblNlbGVjdCl9XHJcbiAgICAgICAgICAgICAgICB2YWx1ZT17cnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXhdfVxyXG4gICAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBhbnkpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihlKTtcclxuICAgICAgICAgICAgICAgICAgc2V0U2VsZWN0KCFvcGVuU2VsZWN0KTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgICBzaG93U2VhcmNoPXt0cnVlfVxyXG4gICAgICAgICAgICAgICAgb3Blbj17b3BlblNlbGVjdH1cclxuICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICB7cnVuTnVtYmVycyAmJlxyXG4gICAgICAgICAgICAgICAgICBydW5OdW1iZXJzLm1hcCgocnVuOiBhbnkpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgPE9wdGlvblxyXG4gICAgICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgc2V0U2VsZWN0KGZhbHNlKTtcclxuICAgICAgICAgICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgICAgICAgICAgdmFsdWU9e3J1bn1cclxuICAgICAgICAgICAgICAgICAgICAgICAga2V5PXtydW4udG9TdHJpbmcoKX1cclxuICAgICAgICAgICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IChcclxuICAgICAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPFNwaW4gLz5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICA8L09wdGlvblBhcmFncmFwaD5cclxuICAgICAgICAgICAgICAgICAgICAgICAgKSA6IChcclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxkaXY+e3J1bn08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICApfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICAgICAgICAgKTtcclxuICAgICAgICAgICAgICAgICAgfSl9XHJcbiAgICAgICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldFJpZ2h0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdfVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XHJcbiAgICA8L0NvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
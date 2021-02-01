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
    if (styledSelectRef.current && styledSelectRef.current.clientWidth) {
      console.log(styledSelectRef.current.clientWidth);
      setStyledSelect(styledSelectRef.current.clientWidth);
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
      lineNumber: 65,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledFormItem"], {
    labelcolor: "white",
    name: 'dataset_name',
    label: "".concat(!withoutLabel ? 'Run' : ''),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 66,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 71,
      columnNumber: 9
    }
  }, !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    disabled: !runNumbers[currentRunNumberIndex - 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 76,
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
      lineNumber: 74,
      columnNumber: 15
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
      columnNumber: 11
    }
  }, __jsx("div", {
    ref: styledSelectRef,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 85,
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
      lineNumber: 86,
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
        lineNumber: 100,
        columnNumber: 23
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 108,
        columnNumber: 27
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 109,
        columnNumber: 29
      }
    })) : __jsx("div", {
      ref: runOptionRef,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 112,
        columnNumber: 29
      }
    }, run));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 121,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 123,
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
      lineNumber: 122,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInJ1bk9wdGlvblJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFJlZiIsInJ1bk9wdGlvbldpZHRoIiwic2V0UnVuT3B0aW9uV2lkdGgiLCJzdHlsZWRTZWxlY3RXaWR0aCIsInNldFN0eWxlZFNlbGVjdCIsInVzZUVmZmVjdCIsImN1cnJlbnQiLCJjbGllbnRXaWR0aCIsImNvbnNvbGUiLCJsb2ciLCJjdXJyZW50UnVuTnVtYmVySW5kZXgiLCJzZXRDdXJyZW50UnVuTnVtYmVySW5kZXgiLCJkYXRhc2V0X25hbWUiLCJ1c2VTZWFyY2giLCJyZXN1bHRzX2dyb3VwZWQiLCJpc0xvYWRpbmciLCJydW5OdW1iZXJzIiwicnVucyIsIm1hcCIsInJ1biIsInRvU3RyaW5nIiwicXVlcnlfcnVuX251bWJlciIsInJ1bl9udW1iZXIiLCJpbmRleE9mIiwiZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUlBO0lBR1FBLE0sR0FBV0MsMkMsQ0FBWEQsTTtBQVlELElBQU1FLFVBQVUsR0FBRyxTQUFiQSxVQUFhLE9BT0g7QUFBQTs7QUFBQSxNQU5yQkMsS0FNcUIsUUFOckJBLEtBTXFCO0FBQUEsTUFMckJDLG1CQUtxQixRQUxyQkEsbUJBS3FCO0FBQUEsTUFKckJDLGFBSXFCLFFBSnJCQSxhQUlxQjtBQUFBLE1BSHJCQyxZQUdxQixRQUhyQkEsWUFHcUI7QUFBQSxNQUZyQkMs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O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BREYsQ0FEUSxHQUtOO0FBQUssU0FBRyxFQUFFakIsWUFBVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQXlCcUIsR0FBekIsQ0FaTixDQURGO0FBaUJELEdBbEJELENBWkosQ0FERixDQURGLENBYkYsRUFpREcsQ0FBQzVCLGFBQUQsSUFDQyxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsUUFBSSxFQUFFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURSO0FBRUUsWUFBUSxFQUFFLENBQUN5QixVQUFVLENBQUNOLHFCQUFxQixHQUFHLENBQXpCLENBRnZCO0FBR0UsUUFBSSxFQUFDLE1BSFA7QUFJRSxXQUFPLEVBQUUsbUJBQU07QUFDYnBCLHlCQUFtQixDQUFDMEIsVUFBVSxDQUFDTixxQkFBcUIsR0FBRyxDQUF6QixDQUFYLENBQW5CO0FBQ0QsS0FOSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FsREosQ0FMRixDQURGLENBREY7QUF3RUQsQ0EvR007O0dBQU10QixVO1VBMEI0QnlCLDBEOzs7S0ExQjVCekIsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4xNTEzNWVhNDgzMTJhZTU1ZGNkZS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QsIHVzZVJlZiB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgQ29sLCBSb3csIFNlbGVjdCwgU3BpbiwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IENhcmV0UmlnaHRGaWxsZWQsIENhcmV0TGVmdEZpbGxlZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkU2VsZWN0LFxyXG4gIE9wdGlvblBhcmFncmFwaCxcclxufSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcblxyXG5jb25zdCB7IE9wdGlvbiB9ID0gU2VsZWN0O1xyXG5cclxuaW50ZXJmYWNlIFJ1bkJyb3dzZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcihjdXJyZW50UnVuTnVtYmVyOiBzdHJpbmcpOiB2b2lkO1xyXG4gIHdpdGhvdXRBcnJvd3M/OiBib29sZWFuO1xyXG4gIHdpdGhvdXRMYWJlbD86IGJvb2xlYW47XHJcbiAgc2VsZWN0b3JXaWR0aD86IHN0cmluZztcclxuICBjdXJyZW50X3J1bl9udW1iZXI/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBSdW5Ccm93c2VyID0gKHtcclxuICBxdWVyeSxcclxuICBzZXRDdXJyZW50UnVuTnVtYmVyLFxyXG4gIHdpdGhvdXRBcnJvd3MsXHJcbiAgd2l0aG91dExhYmVsLFxyXG4gIGN1cnJlbnRfcnVuX251bWJlcixcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZSxcclxufTogUnVuQnJvd3NlclByb3BzKSA9PiB7XHJcbiAgY29uc3QgW29wZW5TZWxlY3QsIHNldFNlbGVjdF0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3QgcnVuT3B0aW9uUmVmID0gdXNlUmVmKG51bGwpXHJcbiAgY29uc3Qgc3R5bGVkU2VsZWN0UmVmID0gdXNlUmVmKG51bGwpXHJcblxyXG4gIGNvbnN0IFtydW5PcHRpb25XaWR0aCwgc2V0UnVuT3B0aW9uV2lkdGhdID0gdXNlU3RhdGUoMClcclxuICBjb25zdCBbc3R5bGVkU2VsZWN0V2lkdGgsIHNldFN0eWxlZFNlbGVjdF0gPSB1c2VTdGF0ZSgwKVxyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50ICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIGNvbnNvbGUubG9nKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKVxyXG4gICAgICBzZXRTdHlsZWRTZWxlY3Qoc3R5bGVkU2VsZWN0UmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpXHJcbiAgICB9XHJcbiAgfSwgW10pXHJcblxyXG4gIGNvbnN0IFtjdXJyZW50UnVuTnVtYmVySW5kZXgsIHNldEN1cnJlbnRSdW5OdW1iZXJJbmRleF0gPSB1c2VTdGF0ZTxudW1iZXI+KDApO1xyXG4gIGNvbnN0IGRhdGFzZXRfbmFtZSA9IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICA/IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICA6IHF1ZXJ5LmRhdGFzZXRfbmFtZTtcclxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgaXNMb2FkaW5nIH0gPSB1c2VTZWFyY2goJycsIGRhdGFzZXRfbmFtZSk7XHJcblxyXG4gIGNvbnN0IHJ1bk51bWJlcnMgPSByZXN1bHRzX2dyb3VwZWRbMF1cclxuICAgID8gcmVzdWx0c19ncm91cGVkWzBdLnJ1bnMubWFwKChydW46IG51bWJlcikgPT4gcnVuLnRvU3RyaW5nKCkpXHJcbiAgICA6IFtdO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgcXVlcnlfcnVuX251bWJlciA9IGN1cnJlbnRfcnVuX251bWJlclxyXG4gICAgICA/IGN1cnJlbnRfcnVuX251bWJlci50b1N0cmluZygpXHJcbiAgICAgIDogcXVlcnkucnVuX251bWJlcjtcclxuICAgIHNldEN1cnJlbnRSdW5OdW1iZXJJbmRleChydW5OdW1iZXJzLmluZGV4T2YocXVlcnlfcnVuX251bWJlcikpO1xyXG4gIH0sIFtydW5OdW1iZXJzLCBpc0xvYWRpbmddKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxDb2w+XHJcbiAgICAgIDxTdHlsZWRGb3JtSXRlbVxyXG4gICAgICAgIGxhYmVsY29sb3I9XCJ3aGl0ZVwiXHJcbiAgICAgICAgbmFtZT17J2RhdGFzZXRfbmFtZSd9XHJcbiAgICAgICAgbGFiZWw9e2AkeyF3aXRob3V0TGFiZWwgPyAnUnVuJyA6ICcnfWB9XHJcbiAgICAgID5cclxuICAgICAgICA8Um93IGp1c3RpZnk9XCJjZW50ZXJcIiBhbGlnbj1cIm1pZGRsZVwiPlxyXG4gICAgICAgICAgeyF3aXRob3V0QXJyb3dzICYmIChcclxuICAgICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgICBkaXNhYmxlZD17IXJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4IC0gMV19XHJcbiAgICAgICAgICAgICAgICBpY29uPXs8Q2FyZXRMZWZ0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4IC0gMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICA8ZGl2IHJlZj17c3R5bGVkU2VsZWN0UmVmfT5cclxuICAgICAgICAgICAgICA8U3R5bGVkU2VsZWN0XHJcbiAgICAgICAgICAgICAgICB3aWR0aD17c3R5bGVkU2VsZWN0V2lkdGggPT09IDAgPyAnZml0LWNvbnRlbnQnOiBzdHlsZWRTZWxlY3RXaWR0aC50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleF19XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKGUpO1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtydW5OdW1iZXJzICYmXHJcbiAgICAgICAgICAgICAgICAgIHJ1bk51bWJlcnMubWFwKChydW46IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZT17cnVufVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e3J1bi50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGRpdiByZWY9e3J1bk9wdGlvblJlZn0+e3J1bn08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICApfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICAgICAgICAgKTtcclxuICAgICAgICAgICAgICAgICAgfSl9XHJcbiAgICAgICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldFJpZ2h0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdfVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XHJcbiAgICA8L0NvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
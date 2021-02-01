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
    width: 100 .toString(),
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
      style: {
        width: "".concat(1000 .toString(), "px")
      },
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInN0eWxlZFNlbGVjdFJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFdpZHRoIiwic2V0U3R5bGVkU2VsZWN0IiwidXNlRWZmZWN0IiwiY3VycmVudCIsImNsaWVudFdpZHRoIiwiY3VycmVudFJ1bk51bWJlckluZGV4Iiwic2V0Q3VycmVudFJ1bk51bWJlckluZGV4IiwiZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwiaXNMb2FkaW5nIiwicnVuTnVtYmVycyIsInJ1bnMiLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsInF1ZXJ5X3J1bl9udW1iZXIiLCJydW5fbnVtYmVyIiwiaW5kZXhPZiIsImUiLCJ3aWR0aCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUlBO0lBR1FBLE0sR0FBV0MsMkMsQ0FBWEQsTTtBQVlELElBQU1FLFVBQVUsR0FBRyxTQUFiQSxVQUFhLE9BT0g7QUFBQTs7QUFBQSxNQU5yQkMsS0FNcUIsUUFOckJBLEtBTXFCO0FBQUEsTUFMckJDLG1CQUtxQixRQUxyQkEsbUJBS3FCO0FBQUEsTUFKckJDLGFBSXFCLFFBSnJCQSxhQUlxQjtBQUFBLE1BSHJCQyxZQUdxQixRQUhyQkEsWUFHcUI7QUFBQSxNQUZyQkMsa0JBRXFCLFFBRnJCQSxrQkFFcUI7QUFBQSxNQURyQkMsb0JBQ3FCLFFBRHJCQSxvQkFDcUI7O0FBQUEs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QUQ7QUFDRSxRQUFJLEVBQUUsTUFBQyxrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BRFI7QUFFRSxZQUFRLEVBQUUsQ0FBQ29CLFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FGdkI7QUFHRSxRQUFJLEVBQUMsTUFIUDtBQUlFLFdBQU8sRUFBRSxtQkFBTTtBQUNiZix5QkFBbUIsQ0FBQ3FCLFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FBWCxDQUFuQjtBQUNELEtBTkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBbERKLENBTEYsQ0FERixDQURGO0FBd0VELENBNUdNOztHQUFNakIsVTtVQXVCNEJvQiwwRDs7O0tBdkI1QnBCLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNTgwYjIwN2RkNzFhZWYzNzJiMjYuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0LCB1c2VSZWYgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IENvbCwgUm93LCBTZWxlY3QsIFNwaW4sIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBDYXJldFJpZ2h0RmlsbGVkLCBDYXJldExlZnRGaWxsZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZFNlbGVjdCxcclxuICBPcHRpb25QYXJhZ3JhcGgsXHJcbn0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi9ob29rcy91c2VTZWFyY2gnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuY29uc3QgeyBPcHRpb24gfSA9IFNlbGVjdDtcclxuXHJcbmludGVyZmFjZSBSdW5Ccm93c2VyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG4gIHNldEN1cnJlbnRSdW5OdW1iZXIoY3VycmVudFJ1bk51bWJlcjogc3RyaW5nKTogdm9pZDtcclxuICB3aXRob3V0QXJyb3dzPzogYm9vbGVhbjtcclxuICB3aXRob3V0TGFiZWw/OiBib29sZWFuO1xyXG4gIHNlbGVjdG9yV2lkdGg/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9ydW5fbnVtYmVyPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgUnVuQnJvd3NlciA9ICh7XHJcbiAgcXVlcnksXHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcixcclxuICB3aXRob3V0QXJyb3dzLFxyXG4gIHdpdGhvdXRMYWJlbCxcclxuICBjdXJyZW50X3J1bl9udW1iZXIsXHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWUsXHJcbn06IFJ1bkJyb3dzZXJQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvcGVuU2VsZWN0LCBzZXRTZWxlY3RdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHN0eWxlZFNlbGVjdFJlZiA9IHVzZVJlZihudWxsKVxyXG5cclxuICBjb25zdCBbc3R5bGVkU2VsZWN0V2lkdGgsIHNldFN0eWxlZFNlbGVjdF0gPSB1c2VTdGF0ZSgwKVxyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50ICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIHNldFN0eWxlZFNlbGVjdChzdHlsZWRTZWxlY3RSZWYuY3VycmVudC5jbGllbnRXaWR0aClcclxuICAgIH1cclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXJJbmRleCwgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4XSA9IHVzZVN0YXRlPG51bWJlcj4oMCk7XHJcbiAgY29uc3QgZGF0YXNldF9uYW1lID0gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgID8gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgIDogcXVlcnkuZGF0YXNldF9uYW1lO1xyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBpc0xvYWRpbmcgfSA9IHVzZVNlYXJjaCgnJywgZGF0YXNldF9uYW1lKTtcclxuXHJcbiAgY29uc3QgcnVuTnVtYmVycyA9IHJlc3VsdHNfZ3JvdXBlZFswXVxyXG4gICAgPyByZXN1bHRzX2dyb3VwZWRbMF0ucnVucy5tYXAoKHJ1bjogbnVtYmVyKSA9PiBydW4udG9TdHJpbmcoKSlcclxuICAgIDogW107XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBxdWVyeV9ydW5fbnVtYmVyID0gY3VycmVudF9ydW5fbnVtYmVyXHJcbiAgICAgID8gY3VycmVudF9ydW5fbnVtYmVyLnRvU3RyaW5nKClcclxuICAgICAgOiBxdWVyeS5ydW5fbnVtYmVyO1xyXG4gICAgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4KHJ1bk51bWJlcnMuaW5kZXhPZihxdWVyeV9ydW5fbnVtYmVyKSk7XHJcbiAgfSwgW3J1bk51bWJlcnMsIGlzTG9hZGluZ10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPENvbD5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtXHJcbiAgICAgICAgbGFiZWxjb2xvcj1cIndoaXRlXCJcclxuICAgICAgICBuYW1lPXsnZGF0YXNldF9uYW1lJ31cclxuICAgICAgICBsYWJlbD17YCR7IXdpdGhvdXRMYWJlbCA/ICdSdW4nIDogJyd9YH1cclxuICAgICAgPlxyXG4gICAgICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXX1cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldExlZnRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxkaXYgcmVmPXtzdHlsZWRTZWxlY3RSZWZ9PlxyXG4gICAgICAgICAgICAgIDxTdHlsZWRTZWxlY3RcclxuICAgICAgICAgICAgICAgIHdpZHRoPXsoMTAwKS50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleF19XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKGUpO1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtydW5OdW1iZXJzICYmXHJcbiAgICAgICAgICAgICAgICAgIHJ1bk51bWJlcnMubWFwKChydW46IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZT17cnVufVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e3J1bi50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGRpdiBzdHlsZT17e3dpZHRoOiBgJHsoMTAwMCkudG9TdHJpbmcoKX1weGB9fT57cnVufTwvZGl2PlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICl9XHJcbiAgICAgICAgICAgICAgICAgICAgICA8L09wdGlvbj5cclxuICAgICAgICAgICAgICAgICAgICApO1xyXG4gICAgICAgICAgICAgICAgICB9KX1cclxuICAgICAgICAgICAgICA8L1N0eWxlZFNlbGVjdD5cclxuICAgICAgICAgICAgPC9kaXY+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICAgICAgaWNvbj17PENhcmV0UmlnaHRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICBkaXNhYmxlZD17IXJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV19XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggKyAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cclxuICAgIDwvQ29sPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
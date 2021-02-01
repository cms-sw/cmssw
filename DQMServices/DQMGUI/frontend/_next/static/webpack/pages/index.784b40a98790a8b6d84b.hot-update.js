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
    width: '42px',
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInN0eWxlZFNlbGVjdFJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFdpZHRoIiwic2V0U3R5bGVkU2VsZWN0IiwidXNlRWZmZWN0IiwiY3VycmVudCIsImNsaWVudFdpZHRoIiwiY3VycmVudFJ1bk51bWJlckluZGV4Iiwic2V0Q3VycmVudFJ1bk51bWJlckluZGV4IiwiZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwiaXNMb2FkaW5nIiwicnVuTnVtYmVycyIsInJ1bnMiLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsInF1ZXJ5X3J1bl9udW1iZXIiLCJydW5fbnVtYmVyIiwiaW5kZXhPZiIsImUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFJQTtJQUdRQSxNLEdBQVdDLDJDLENBQVhELE07QUFZRCxJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQU9IO0FBQUE7O0FBQUEsTUFOckJDLEtBTXFCLFFBTnJCQSxLQU1xQjtBQUFBLE1BTHJCQyxtQkFLcUIsUUFMckJBLG1CQUtxQjtBQUFBLE1BSnJCQyxhQUlxQixRQUpyQkEsYUFJcUI7QUFBQSxNQUhyQkMsWUFHcUIsUUFIckJBLFlBR3FCO0FBQUEsTUFGckJDLGtCQUVxQixRQUZyQkEsa0JBRXFCO0FBQUEsTUFEckJDLG9CQUNxQixRQURyQkEsb0JBQ3FCOztBQUFBLGtCQUNXQyxz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LFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FGdkI7QUFHRSxRQUFJLEVBQUMsTUFIUDtBQUlFLFdBQU8sRUFBRSxtQkFBTTtBQUNiZix5QkFBbUIsQ0FBQ3FCLFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FBWCxDQUFuQjtBQUNELEtBTkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBbERKLENBTEYsQ0FERixDQURGO0FBd0VELENBNUdNOztHQUFNakIsVTtVQXVCNEJvQiwwRDs7O0tBdkI1QnBCLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNzg0YjQwYTk4NzkwYThiNmQ4NGIuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0LCB1c2VSZWYgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IENvbCwgUm93LCBTZWxlY3QsIFNwaW4sIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBDYXJldFJpZ2h0RmlsbGVkLCBDYXJldExlZnRGaWxsZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZFNlbGVjdCxcclxuICBPcHRpb25QYXJhZ3JhcGgsXHJcbn0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi9ob29rcy91c2VTZWFyY2gnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuY29uc3QgeyBPcHRpb24gfSA9IFNlbGVjdDtcclxuXHJcbmludGVyZmFjZSBSdW5Ccm93c2VyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG4gIHNldEN1cnJlbnRSdW5OdW1iZXIoY3VycmVudFJ1bk51bWJlcjogc3RyaW5nKTogdm9pZDtcclxuICB3aXRob3V0QXJyb3dzPzogYm9vbGVhbjtcclxuICB3aXRob3V0TGFiZWw/OiBib29sZWFuO1xyXG4gIHNlbGVjdG9yV2lkdGg/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9ydW5fbnVtYmVyPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgUnVuQnJvd3NlciA9ICh7XHJcbiAgcXVlcnksXHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcixcclxuICB3aXRob3V0QXJyb3dzLFxyXG4gIHdpdGhvdXRMYWJlbCxcclxuICBjdXJyZW50X3J1bl9udW1iZXIsXHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWUsXHJcbn06IFJ1bkJyb3dzZXJQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvcGVuU2VsZWN0LCBzZXRTZWxlY3RdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHN0eWxlZFNlbGVjdFJlZiA9IHVzZVJlZihudWxsKVxyXG5cclxuICBjb25zdCBbc3R5bGVkU2VsZWN0V2lkdGgsIHNldFN0eWxlZFNlbGVjdF0gPSB1c2VTdGF0ZSgwKVxyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50ICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIHNldFN0eWxlZFNlbGVjdChzdHlsZWRTZWxlY3RSZWYuY3VycmVudC5jbGllbnRXaWR0aClcclxuICAgIH1cclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXJJbmRleCwgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4XSA9IHVzZVN0YXRlPG51bWJlcj4oMCk7XHJcbiAgY29uc3QgZGF0YXNldF9uYW1lID0gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgID8gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgIDogcXVlcnkuZGF0YXNldF9uYW1lO1xyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBpc0xvYWRpbmcgfSA9IHVzZVNlYXJjaCgnJywgZGF0YXNldF9uYW1lKTtcclxuXHJcbiAgY29uc3QgcnVuTnVtYmVycyA9IHJlc3VsdHNfZ3JvdXBlZFswXVxyXG4gICAgPyByZXN1bHRzX2dyb3VwZWRbMF0ucnVucy5tYXAoKHJ1bjogbnVtYmVyKSA9PiBydW4udG9TdHJpbmcoKSlcclxuICAgIDogW107XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBxdWVyeV9ydW5fbnVtYmVyID0gY3VycmVudF9ydW5fbnVtYmVyXHJcbiAgICAgID8gY3VycmVudF9ydW5fbnVtYmVyLnRvU3RyaW5nKClcclxuICAgICAgOiBxdWVyeS5ydW5fbnVtYmVyO1xyXG4gICAgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4KHJ1bk51bWJlcnMuaW5kZXhPZihxdWVyeV9ydW5fbnVtYmVyKSk7XHJcbiAgfSwgW3J1bk51bWJlcnMsIGlzTG9hZGluZ10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPENvbD5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtXHJcbiAgICAgICAgbGFiZWxjb2xvcj1cIndoaXRlXCJcclxuICAgICAgICBuYW1lPXsnZGF0YXNldF9uYW1lJ31cclxuICAgICAgICBsYWJlbD17YCR7IXdpdGhvdXRMYWJlbCA/ICdSdW4nIDogJyd9YH1cclxuICAgICAgPlxyXG4gICAgICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXX1cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldExlZnRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxkaXYgcmVmPXtzdHlsZWRTZWxlY3RSZWZ9PlxyXG4gICAgICAgICAgICAgIDxTdHlsZWRTZWxlY3RcclxuICAgICAgICAgICAgICAgIHdpZHRoPXsnNDJweCd9XHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpfVxyXG4gICAgICAgICAgICAgICAgdmFsdWU9e3J1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4XX1cclxuICAgICAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIoZSk7XHJcbiAgICAgICAgICAgICAgICAgIHNldFNlbGVjdCghb3BlblNlbGVjdCk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgc2hvd1NlYXJjaD17dHJ1ZX1cclxuICAgICAgICAgICAgICAgIG9wZW49e29wZW5TZWxlY3R9XHJcbiAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAge3J1bk51bWJlcnMgJiZcclxuICAgICAgICAgICAgICAgICAgcnVuTnVtYmVycy5tYXAoKHJ1bjogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25cclxuICAgICAgICAgICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIHNldFNlbGVjdChmYWxzZSk7XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhbHVlPXtydW59XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIGtleT17cnVuLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgIHtpc0xvYWRpbmcgPyAoXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgPE9wdGlvblBhcmFncmFwaD5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIDxTcGluIC8+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgPC9PcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8ZGl2PntydW59PC9kaXY+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgKX1cclxuICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uPlxyXG4gICAgICAgICAgICAgICAgICAgICk7XHJcbiAgICAgICAgICAgICAgICAgIH0pfVxyXG4gICAgICAgICAgICAgIDwvU3R5bGVkU2VsZWN0PlxyXG4gICAgICAgICAgICA8L2Rpdj5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgeyF3aXRob3V0QXJyb3dzICYmIChcclxuICAgICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgICBpY29uPXs8Q2FyZXRSaWdodEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggKyAxXX1cclxuICAgICAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0Q3VycmVudFJ1bk51bWJlcihydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdKTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICApfVxyXG4gICAgICAgIDwvUm93PlxyXG4gICAgICA8L1N0eWxlZEZvcm1JdGVtPlxyXG4gICAgPC9Db2w+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
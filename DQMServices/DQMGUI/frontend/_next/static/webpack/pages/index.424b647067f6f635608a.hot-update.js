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
    if (runOptionRef && runOptionRef.current.clientWidth) {
      setStyledSelect(runOptionRef.current.clientWidth);
    } else if (runOptionRef && styledSelectRef.current.clientWidth) {
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
      lineNumber: 67,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledFormItem"], {
    labelcolor: "white",
    name: 'dataset_name',
    label: "".concat(!withoutLabel ? 'Run' : ''),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 9
    }
  }, !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    disabled: !runNumbers[currentRunNumberIndex - 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 78,
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
      lineNumber: 76,
      columnNumber: 15
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 86,
      columnNumber: 11
    }
  }, __jsx("div", {
    ref: styledSelectRef,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 87,
      columnNumber: 13
    }
  }, __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSelect"], {
    width: styledSelectWidth.toString(),
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
      lineNumber: 88,
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
        lineNumber: 102,
        columnNumber: 23
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_4__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 110,
        columnNumber: 27
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 111,
        columnNumber: 29
      }
    })) : __jsx("div", {
      ref: runOptionRef,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 114,
        columnNumber: 29
      }
    }, run));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 123,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 125,
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
      lineNumber: 124,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInJ1bk9wdGlvblJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFJlZiIsInJ1bk9wdGlvbldpZHRoIiwic2V0UnVuT3B0aW9uV2lkdGgiLCJzdHlsZWRTZWxlY3RXaWR0aCIsInNldFN0eWxlZFNlbGVjdCIsInVzZUVmZmVjdCIsImN1cnJlbnQiLCJjbGllbnRXaWR0aCIsImN1cnJlbnRSdW5OdW1iZXJJbmRleCIsInNldEN1cnJlbnRSdW5OdW1iZXJJbmRleCIsImRhdGFzZXRfbmFtZSIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsInJ1bk51bWJlcnMiLCJydW5zIiwibWFwIiwicnVuIiwidG9TdHJpbmciLCJxdWVyeV9ydW5fbnVtYmVyIiwicnVuX251bWJlciIsImluZGV4T2YiLCJlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBSUE7SUFHUUEsTSxHQUFXQywyQyxDQUFYRCxNO0FBWUQsSUFBTUUsVUFBVSxHQUFHLFNBQWJBLFVBQWEsT0FPSDtBQUFBOztBQUFBLE1BTnJCQyxLQU1xQixRQU5yQkEsS0FNcUI7QUFBQSxNQUxyQkMsbUJBS3FCLFFBTHJCQSxtQkFLcUI7QUFBQSxNQUpyQkMsYUFJcUIsUUFKckJBLGFBSXFCO0FBQUEsTUFIckJDLFlBR3FCLFFBSHJCQSxZQUdxQjtBQUFBLE1BRnJCQyxrQkFFcUIsUUFGckJBLGtC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QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERixDQURRLEdBS047QUFBSyxTQUFHLEVBQUVmLFlBQVY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUF5Qm1CLEdBQXpCLENBWk4sQ0FERjtBQWlCRCxHQWxCRCxDQVpKLENBREYsQ0FERixDQWJGLEVBaURHLENBQUMxQixhQUFELElBQ0MsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUNFLFFBQUksRUFBRSxNQUFDLGtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEUjtBQUVFLFlBQVEsRUFBRSxDQUFDdUIsVUFBVSxDQUFDTixxQkFBcUIsR0FBRyxDQUF6QixDQUZ2QjtBQUdFLFFBQUksRUFBQyxNQUhQO0FBSUUsV0FBTyxFQUFFLG1CQUFNO0FBQ2JsQix5QkFBbUIsQ0FBQ3dCLFVBQVUsQ0FBQ04scUJBQXFCLEdBQUcsQ0FBekIsQ0FBWCxDQUFuQjtBQUNELEtBTkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBbERKLENBTEYsQ0FERixDQURGO0FBd0VELENBakhNOztHQUFNcEIsVTtVQTRCNEJ1QiwwRDs7O0tBNUI1QnZCLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDI0YjY0NzA2N2Y2ZjYzNTYwOGEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0LCB1c2VSZWYgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IENvbCwgUm93LCBTZWxlY3QsIFNwaW4sIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBDYXJldFJpZ2h0RmlsbGVkLCBDYXJldExlZnRGaWxsZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQge1xyXG4gIFN0eWxlZFNlbGVjdCxcclxuICBPcHRpb25QYXJhZ3JhcGgsXHJcbn0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi9ob29rcy91c2VTZWFyY2gnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuY29uc3QgeyBPcHRpb24gfSA9IFNlbGVjdDtcclxuXHJcbmludGVyZmFjZSBSdW5Ccm93c2VyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG4gIHNldEN1cnJlbnRSdW5OdW1iZXIoY3VycmVudFJ1bk51bWJlcjogc3RyaW5nKTogdm9pZDtcclxuICB3aXRob3V0QXJyb3dzPzogYm9vbGVhbjtcclxuICB3aXRob3V0TGFiZWw/OiBib29sZWFuO1xyXG4gIHNlbGVjdG9yV2lkdGg/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9ydW5fbnVtYmVyPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgUnVuQnJvd3NlciA9ICh7XHJcbiAgcXVlcnksXHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcixcclxuICB3aXRob3V0QXJyb3dzLFxyXG4gIHdpdGhvdXRMYWJlbCxcclxuICBjdXJyZW50X3J1bl9udW1iZXIsXHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWUsXHJcbn06IFJ1bkJyb3dzZXJQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvcGVuU2VsZWN0LCBzZXRTZWxlY3RdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHJ1bk9wdGlvblJlZiA9IHVzZVJlZihudWxsKVxyXG4gIGNvbnN0IHN0eWxlZFNlbGVjdFJlZiA9IHVzZVJlZihudWxsKVxyXG5cclxuICBjb25zdCBbcnVuT3B0aW9uV2lkdGgsIHNldFJ1bk9wdGlvbldpZHRoXSA9IHVzZVN0YXRlKDApXHJcbiAgY29uc3QgW3N0eWxlZFNlbGVjdFdpZHRoLCBzZXRTdHlsZWRTZWxlY3RdID0gdXNlU3RhdGUoMClcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGlmIChydW5PcHRpb25SZWYgJiYgcnVuT3B0aW9uUmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpIHtcclxuICAgICAgc2V0U3R5bGVkU2VsZWN0KHJ1bk9wdGlvblJlZi5jdXJyZW50LmNsaWVudFdpZHRoKVxyXG4gICAgfVxyXG4gICAgZWxzZSBpZiAocnVuT3B0aW9uUmVmICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIHNldFN0eWxlZFNlbGVjdChzdHlsZWRTZWxlY3RSZWYuY3VycmVudC5jbGllbnRXaWR0aClcclxuICAgIH1cclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXJJbmRleCwgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4XSA9IHVzZVN0YXRlPG51bWJlcj4oMCk7XHJcbiAgY29uc3QgZGF0YXNldF9uYW1lID0gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgID8gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgIDogcXVlcnkuZGF0YXNldF9uYW1lO1xyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBpc0xvYWRpbmcgfSA9IHVzZVNlYXJjaCgnJywgZGF0YXNldF9uYW1lKTtcclxuXHJcbiAgY29uc3QgcnVuTnVtYmVycyA9IHJlc3VsdHNfZ3JvdXBlZFswXVxyXG4gICAgPyByZXN1bHRzX2dyb3VwZWRbMF0ucnVucy5tYXAoKHJ1bjogbnVtYmVyKSA9PiBydW4udG9TdHJpbmcoKSlcclxuICAgIDogW107XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBxdWVyeV9ydW5fbnVtYmVyID0gY3VycmVudF9ydW5fbnVtYmVyXHJcbiAgICAgID8gY3VycmVudF9ydW5fbnVtYmVyLnRvU3RyaW5nKClcclxuICAgICAgOiBxdWVyeS5ydW5fbnVtYmVyO1xyXG4gICAgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4KHJ1bk51bWJlcnMuaW5kZXhPZihxdWVyeV9ydW5fbnVtYmVyKSk7XHJcbiAgfSwgW3J1bk51bWJlcnMsIGlzTG9hZGluZ10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPENvbD5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtXHJcbiAgICAgICAgbGFiZWxjb2xvcj1cIndoaXRlXCJcclxuICAgICAgICBuYW1lPXsnZGF0YXNldF9uYW1lJ31cclxuICAgICAgICBsYWJlbD17YCR7IXdpdGhvdXRMYWJlbCA/ICdSdW4nIDogJyd9YH1cclxuICAgICAgPlxyXG4gICAgICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXX1cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldExlZnRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxkaXYgcmVmPXtzdHlsZWRTZWxlY3RSZWZ9PlxyXG4gICAgICAgICAgICAgIDxTdHlsZWRTZWxlY3RcclxuICAgICAgICAgICAgICAgIHdpZHRoPXtzdHlsZWRTZWxlY3RXaWR0aC50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleF19XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKGUpO1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtydW5OdW1iZXJzICYmXHJcbiAgICAgICAgICAgICAgICAgIHJ1bk51bWJlcnMubWFwKChydW46IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZT17cnVufVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e3J1bi50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGRpdiByZWY9e3J1bk9wdGlvblJlZn0+e3J1bn08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICApfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICAgICAgICAgKTtcclxuICAgICAgICAgICAgICAgICAgfSl9XHJcbiAgICAgICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldFJpZ2h0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdfVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XHJcbiAgICA8L0NvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
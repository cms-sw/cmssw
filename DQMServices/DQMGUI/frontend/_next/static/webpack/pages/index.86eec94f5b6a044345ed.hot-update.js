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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9ydW5zQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiUnVuQnJvd3NlciIsInF1ZXJ5Iiwic2V0Q3VycmVudFJ1bk51bWJlciIsIndpdGhvdXRBcnJvd3MiLCJ3aXRob3V0TGFiZWwiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJjdXJyZW50X2RhdGFzZXRfbmFtZSIsInVzZVN0YXRlIiwib3BlblNlbGVjdCIsInNldFNlbGVjdCIsInJ1bk9wdGlvblJlZiIsInVzZVJlZiIsInN0eWxlZFNlbGVjdFJlZiIsInJ1bk9wdGlvbldpZHRoIiwic2V0UnVuT3B0aW9uV2lkdGgiLCJzdHlsZWRTZWxlY3RXaWR0aCIsInNldFN0eWxlZFNlbGVjdCIsInVzZUVmZmVjdCIsImN1cnJlbnQiLCJjbGllbnRXaWR0aCIsImN1cnJlbnRSdW5OdW1iZXJJbmRleCIsInNldEN1cnJlbnRSdW5OdW1iZXJJbmRleCIsImRhdGFzZXRfbmFtZSIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsInJ1bk51bWJlcnMiLCJydW5zIiwibWFwIiwicnVuIiwidG9TdHJpbmciLCJxdWVyeV9ydW5fbnVtYmVyIiwicnVuX251bWJlciIsImluZGV4T2YiLCJlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBSUE7SUFHUUEsTSxHQUFXQywyQyxDQUFYRCxNO0FBWUQsSUFBTUUsVUFBVSxHQUFHLFNBQWJBLFVBQWEsT0FPSDtBQUFBOztBQUFBLE1BTnJCQyxLQU1xQixRQU5yQkEsS0FNcUI7QUFBQSxNQUxyQkMsbUJBS3FCLFFBTHJCQSxtQkFLcUI7QUFBQSxNQUpyQkMsYUFJcUIsUUFKckJBLGFBSXFCO0FBQUEsTUFIckJDLFlBR3FCLFFBSHJCQSxZQUdxQjtBQUFBLE1BRnJCQyxrQkFFcUIsUUFGckJBLGtC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Q0FaTixDQURGO0FBaUJELEdBbEJELENBWkosQ0FERixDQURGLENBYkYsRUFpREcsQ0FBQzFCLGFBQUQsSUFDQyxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsUUFBSSxFQUFFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURSO0FBRUUsWUFBUSxFQUFFLENBQUN1QixVQUFVLENBQUNOLHFCQUFxQixHQUFHLENBQXpCLENBRnZCO0FBR0UsUUFBSSxFQUFDLE1BSFA7QUFJRSxXQUFPLEVBQUUsbUJBQU07QUFDYmxCLHlCQUFtQixDQUFDd0IsVUFBVSxDQUFDTixxQkFBcUIsR0FBRyxDQUF6QixDQUFYLENBQW5CO0FBQ0QsS0FOSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FsREosQ0FMRixDQURGLENBREY7QUF3RUQsQ0E5R007O0dBQU1wQixVO1VBeUI0QnVCLDBEOzs7S0F6QjVCdkIsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC44NmVlYzk0ZjViNmEwNDQzNDVlZC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QsIHVzZVJlZiB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgQ29sLCBSb3csIFNlbGVjdCwgU3BpbiwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IENhcmV0UmlnaHRGaWxsZWQsIENhcmV0TGVmdEZpbGxlZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkU2VsZWN0LFxyXG4gIE9wdGlvblBhcmFncmFwaCxcclxufSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcblxyXG5jb25zdCB7IE9wdGlvbiB9ID0gU2VsZWN0O1xyXG5cclxuaW50ZXJmYWNlIFJ1bkJyb3dzZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbiAgc2V0Q3VycmVudFJ1bk51bWJlcihjdXJyZW50UnVuTnVtYmVyOiBzdHJpbmcpOiB2b2lkO1xyXG4gIHdpdGhvdXRBcnJvd3M/OiBib29sZWFuO1xyXG4gIHdpdGhvdXRMYWJlbD86IGJvb2xlYW47XHJcbiAgc2VsZWN0b3JXaWR0aD86IHN0cmluZztcclxuICBjdXJyZW50X3J1bl9udW1iZXI/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBSdW5Ccm93c2VyID0gKHtcclxuICBxdWVyeSxcclxuICBzZXRDdXJyZW50UnVuTnVtYmVyLFxyXG4gIHdpdGhvdXRBcnJvd3MsXHJcbiAgd2l0aG91dExhYmVsLFxyXG4gIGN1cnJlbnRfcnVuX251bWJlcixcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZSxcclxufTogUnVuQnJvd3NlclByb3BzKSA9PiB7XHJcbiAgY29uc3QgW29wZW5TZWxlY3QsIHNldFNlbGVjdF0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3QgcnVuT3B0aW9uUmVmID0gdXNlUmVmKG51bGwpXHJcbiAgY29uc3Qgc3R5bGVkU2VsZWN0UmVmID0gdXNlUmVmKG51bGwpXHJcblxyXG4gIGNvbnN0IFtydW5PcHRpb25XaWR0aCwgc2V0UnVuT3B0aW9uV2lkdGhdID0gdXNlU3RhdGUoMClcclxuICBjb25zdCBbc3R5bGVkU2VsZWN0V2lkdGgsIHNldFN0eWxlZFNlbGVjdF0gPSB1c2VTdGF0ZSgwKVxyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50ICYmIHN0eWxlZFNlbGVjdFJlZi5jdXJyZW50LmNsaWVudFdpZHRoKSB7XHJcbiAgICAgIHNldFN0eWxlZFNlbGVjdChzdHlsZWRTZWxlY3RSZWYuY3VycmVudC5jbGllbnRXaWR0aClcclxuICAgIH1cclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXJJbmRleCwgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4XSA9IHVzZVN0YXRlPG51bWJlcj4oMCk7XHJcbiAgY29uc3QgZGF0YXNldF9uYW1lID0gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgID8gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgIDogcXVlcnkuZGF0YXNldF9uYW1lO1xyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBpc0xvYWRpbmcgfSA9IHVzZVNlYXJjaCgnJywgZGF0YXNldF9uYW1lKTtcclxuXHJcbiAgY29uc3QgcnVuTnVtYmVycyA9IHJlc3VsdHNfZ3JvdXBlZFswXVxyXG4gICAgPyByZXN1bHRzX2dyb3VwZWRbMF0ucnVucy5tYXAoKHJ1bjogbnVtYmVyKSA9PiBydW4udG9TdHJpbmcoKSlcclxuICAgIDogW107XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBxdWVyeV9ydW5fbnVtYmVyID0gY3VycmVudF9ydW5fbnVtYmVyXHJcbiAgICAgID8gY3VycmVudF9ydW5fbnVtYmVyLnRvU3RyaW5nKClcclxuICAgICAgOiBxdWVyeS5ydW5fbnVtYmVyO1xyXG4gICAgc2V0Q3VycmVudFJ1bk51bWJlckluZGV4KHJ1bk51bWJlcnMuaW5kZXhPZihxdWVyeV9ydW5fbnVtYmVyKSk7XHJcbiAgfSwgW3J1bk51bWJlcnMsIGlzTG9hZGluZ10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPENvbD5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtXHJcbiAgICAgICAgbGFiZWxjb2xvcj1cIndoaXRlXCJcclxuICAgICAgICBuYW1lPXsnZGF0YXNldF9uYW1lJ31cclxuICAgICAgICBsYWJlbD17YCR7IXdpdGhvdXRMYWJlbCA/ICdSdW4nIDogJyd9YH1cclxuICAgICAgPlxyXG4gICAgICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGRpc2FibGVkPXshcnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXX1cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldExlZnRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldEN1cnJlbnRSdW5OdW1iZXIocnVuTnVtYmVyc1tjdXJyZW50UnVuTnVtYmVySW5kZXggLSAxXSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxkaXYgcmVmPXtzdHlsZWRTZWxlY3RSZWZ9PlxyXG4gICAgICAgICAgICAgIDxTdHlsZWRTZWxlY3RcclxuICAgICAgICAgICAgICAgIHdpZHRoPXtzdHlsZWRTZWxlY3RXaWR0aC50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleF19XHJcbiAgICAgICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKGUpO1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtydW5OdW1iZXJzICYmXHJcbiAgICAgICAgICAgICAgICAgIHJ1bk51bWJlcnMubWFwKChydW46IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgICAgICAgICB2YWx1ZT17cnVufVxyXG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e3J1bi50b1N0cmluZygpfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgPGRpdiByZWY9e3J1bk9wdGlvblJlZn0+e3J1bn08L2Rpdj5cclxuICAgICAgICAgICAgICAgICAgICAgICAgICApfVxyXG4gICAgICAgICAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICAgICAgICAgKTtcclxuICAgICAgICAgICAgICAgICAgfSl9XHJcbiAgICAgICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgICAgIDwvZGl2PlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgICAgIGljb249ezxDYXJldFJpZ2h0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFydW5OdW1iZXJzW2N1cnJlbnRSdW5OdW1iZXJJbmRleCArIDFdfVxyXG4gICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRDdXJyZW50UnVuTnVtYmVyKHJ1bk51bWJlcnNbY3VycmVudFJ1bk51bWJlckluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XHJcbiAgICA8L0NvbD5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9
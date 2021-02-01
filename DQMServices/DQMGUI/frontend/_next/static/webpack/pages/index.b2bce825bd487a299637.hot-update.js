webpackHotUpdate_N_E("pages/index",{

/***/ "./components/browsing/datasetsBrowsing/datasetsBrowser.tsx":
/*!******************************************************************!*\
  !*** ./components/browsing/datasetsBrowsing/datasetsBrowser.tsx ***!
  \******************************************************************/
/*! exports provided: DatasetsBrowser */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DatasetsBrowser", function() { return DatasetsBrowser; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _hooks_useSearch__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../hooks/useSearch */ "./hooks/useSearch.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../styledComponents */ "./components/styledComponents.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/browsing/datasetsBrowsing/datasetsBrowser.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;






var Option = antd__WEBPACK_IMPORTED_MODULE_1__["Select"].Option;
var DatasetsBrowser = function DatasetsBrowser(_ref) {
  _s();

  var withoutArrows = _ref.withoutArrows,
      setCurrentDataset = _ref.setCurrentDataset,
      selectorWidth = _ref.selectorWidth,
      query = _ref.query,
      current_dataset_name = _ref.current_dataset_name,
      current_run_number = _ref.current_run_number;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(false),
      openSelect = _useState[0],
      setSelect = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(0),
      currentDatasetNameIndex = _useState2[0],
      setCurrentDatasetNameIndex = _useState2[1];

  var run_number = current_run_number ? current_run_number : query.run_number;

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_4__["useSearch"])(run_number, ''),
      results_grouped = _useSearch.results_grouped,
      isLoading = _useSearch.isLoading;

  var datasets = results_grouped.map(function (result) {
    return result.dataset;
  });
  Object(react__WEBPACK_IMPORTED_MODULE_0__["useEffect"])(function () {
    var query_dataset = current_dataset_name ? current_dataset_name : query.dataset_name;
    setCurrentDatasetNameIndex(datasets.indexOf(query_dataset));
  }, [isLoading, datasets]);
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 5
    }
  }, !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    disabled: !datasets[currentDatasetNameIndex - 1],
    type: "link",
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 57,
        columnNumber: 19
      }
    }),
    onClick: function onClick() {
      setCurrentDataset(datasets[currentDatasetNameIndex - 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["CustomCol"], {
    width: selectorWidth,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 64,
      columnNumber: 7
    }
  }, __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 65,
      columnNumber: 9
    }
  }, __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledSelect"], {
    onChange: function onChange(e) {
      setCurrentDataset(e);
    },
    style: {
      minWidth: '200px'
    },
    value: datasets[currentDatasetNameIndex],
    dropdownMatchSelectWidth: false,
    onClick: function onClick() {
      return setSelect(!openSelect);
    },
    open: openSelect,
    showSearch: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 66,
      columnNumber: 11
    }
  }, results_grouped.map(function (result) {
    return __jsx(Option, {
      onClick: function onClick() {
        setSelect(false);
      },
      value: result.dataset,
      key: result.dataset,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 78,
        columnNumber: 15
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_3__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 86,
        columnNumber: 19
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 87,
        columnNumber: 21
      }
    })) : __jsx("p", {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 90,
        columnNumber: 19
      }
    }, result.dataset));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    disabled: !datasets[currentDatasetNameIndex + 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 102,
        columnNumber: 19
      }
    }),
    onClick: function onClick() {
      setCurrentDataset(datasets[currentDatasetNameIndex + 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 11
    }
  })));
};

_s(DatasetsBrowser, "j0sLjf1EgPcAjI6EpAzcQwCV8P8=", false, function () {
  return [_hooks_useSearch__WEBPACK_IMPORTED_MODULE_4__["useSearch"]];
});

_c = DatasetsBrowser;

var _c;

$RefreshReg$(_c, "DatasetsBrowser");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXRzQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiRGF0YXNldHNCcm93c2VyIiwid2l0aG91dEFycm93cyIsInNldEN1cnJlbnREYXRhc2V0Iiwic2VsZWN0b3JXaWR0aCIsInF1ZXJ5IiwiY3VycmVudF9kYXRhc2V0X25hbWUiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJ1c2VTdGF0ZSIsIm9wZW5TZWxlY3QiLCJzZXRTZWxlY3QiLCJjdXJyZW50RGF0YXNldE5hbWVJbmRleCIsInNldEN1cnJlbnREYXRhc2V0TmFtZUluZGV4IiwicnVuX251bWJlciIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsImRhdGFzZXRzIiwibWFwIiwicmVzdWx0IiwiZGF0YXNldCIsInVzZUVmZmVjdCIsInF1ZXJ5X2RhdGFzZXQiLCJkYXRhc2V0X25hbWUiLCJpbmRleE9mIiwiZSIsIm1pbldpZHRoIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQUlBO0FBRUE7SUFXUUEsTSxHQUFXQywyQyxDQUFYRCxNO0FBRUQsSUFBTUUsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixPQU9IO0FBQUE7O0FBQUEsTUFOMUJDLGFBTTBCLFFBTjFCQSxhQU0wQjtBQUFBLE1BTDFCQyxpQkFLMEIsUUFMMUJBLGlCQUswQjtBQUFBLE1BSjFCQyxhQUkwQixRQUoxQkEsYUFJMEI7QUFBQSxNQUgxQkMsS0FHMEIsUUFIMUJBLEtBRzBCO0FBQUEsTUFGMUJDLG9CQUUwQixRQUYxQkEsb0JBRTBCO0FBQUEsTUFEMUJDLGtCQUMwQixRQUQxQkEsa0JBQzBCOztBQUFBLGtCQUNNQyxzREFBUSxDQUFDLEtBQUQsQ0FEZDtBQUFBLE1BQ25CQyxVQURtQjtBQUFBLE1BQ1BDLFNBRE87O0FB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QUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBSUcsTUFBTSxDQUFDQyxPQUFYLENBWkosQ0FEbUI7QUFBQSxHQUFwQixDQVhILENBREYsQ0FERixDQWJGLEVBOENHLENBQUNsQixhQUFELElBQ0MsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUNFLFFBQUksRUFBQyxNQURQO0FBRUUsWUFBUSxFQUFFLENBQUNlLFFBQVEsQ0FBQ04sdUJBQXVCLEdBQUcsQ0FBM0IsQ0FGckI7QUFHRSxRQUFJLEVBQUUsTUFBQyxrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BSFI7QUFJRSxXQUFPLEVBQUUsbUJBQU07QUFDYlIsdUJBQWlCLENBQUNjLFFBQVEsQ0FBQ04sdUJBQXVCLEdBQUcsQ0FBM0IsQ0FBVCxDQUFqQjtBQUNELEtBTkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBL0NKLENBREY7QUE2REQsQ0F2Rk07O0dBQU1WLGU7VUFhNEJhLDBEOzs7S0FiNUJiLGUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguYjJiY2U4MjViZDQ4N2EyOTk2MzcuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VTdGF0ZSwgdXNlRWZmZWN0IH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBDb2wsIFNlbGVjdCwgUm93LCBTcGluLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgQ2FyZXRSaWdodEZpbGxlZCwgQ2FyZXRMZWZ0RmlsbGVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHtcclxuICBTdHlsZWRTZWxlY3QsXHJcbiAgT3B0aW9uUGFyYWdyYXBoLFxyXG59IGZyb20gJy4uLy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdXNlU2VhcmNoIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlU2VhcmNoJztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgQ3VzdG9tQ29sIH0gZnJvbSAnLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcblxyXG5pbnRlcmZhY2UgRGF0YXNldHNCcm93c2VyUHJvcHMge1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG4gIHNldEN1cnJlbnREYXRhc2V0KGN1cnJlbnREYXRhc2V0OiBzdHJpbmcpOiB2b2lkO1xyXG4gIHdpdGhvdXRBcnJvd3M/OiBib29sZWFuO1xyXG4gIHNlbGVjdG9yV2lkdGg/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XHJcbiAgY3VycmVudF9ydW5fbnVtYmVyPzogc3RyaW5nO1xyXG59XHJcblxyXG5jb25zdCB7IE9wdGlvbiB9ID0gU2VsZWN0O1xyXG5cclxuZXhwb3J0IGNvbnN0IERhdGFzZXRzQnJvd3NlciA9ICh7XHJcbiAgd2l0aG91dEFycm93cyxcclxuICBzZXRDdXJyZW50RGF0YXNldCxcclxuICBzZWxlY3RvcldpZHRoLFxyXG4gIHF1ZXJ5LFxyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lLFxyXG4gIGN1cnJlbnRfcnVuX251bWJlcixcclxufTogRGF0YXNldHNCcm93c2VyUHJvcHMpID0+IHtcclxuICBjb25zdCBbb3BlblNlbGVjdCwgc2V0U2VsZWN0XSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBbY3VycmVudERhdGFzZXROYW1lSW5kZXgsIHNldEN1cnJlbnREYXRhc2V0TmFtZUluZGV4XSA9IHVzZVN0YXRlPFxyXG4gICAgbnVtYmVyXHJcbiAgPigwKTtcclxuICBjb25zdCBydW5fbnVtYmVyID0gY3VycmVudF9ydW5fbnVtYmVyID8gY3VycmVudF9ydW5fbnVtYmVyIDogcXVlcnkucnVuX251bWJlcjtcclxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgaXNMb2FkaW5nIH0gPSB1c2VTZWFyY2gocnVuX251bWJlciwgJycpO1xyXG5cclxuICBjb25zdCBkYXRhc2V0cyA9IHJlc3VsdHNfZ3JvdXBlZC5tYXAoKHJlc3VsdCkgPT4ge1xyXG4gICAgcmV0dXJuIHJlc3VsdC5kYXRhc2V0O1xyXG4gIH0pO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgcXVlcnlfZGF0YXNldCA9IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICAgID8gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgICAgOiBxdWVyeS5kYXRhc2V0X25hbWU7XHJcbiAgICBzZXRDdXJyZW50RGF0YXNldE5hbWVJbmRleChkYXRhc2V0cy5pbmRleE9mKHF1ZXJ5X2RhdGFzZXQpKTtcclxuICB9LCBbaXNMb2FkaW5nLCBkYXRhc2V0c10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFJvdyBqdXN0aWZ5PVwiY2VudGVyXCIgYWxpZ249XCJtaWRkbGVcIj5cclxuICAgICAgeyF3aXRob3V0QXJyb3dzICYmIChcclxuICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICBkaXNhYmxlZD17IWRhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4IC0gMV19XHJcbiAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgaWNvbj17PENhcmV0TGVmdEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgIHNldEN1cnJlbnREYXRhc2V0KGRhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4IC0gMV0pO1xyXG4gICAgICAgICAgICB9fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbD5cclxuICAgICAgKX1cclxuICAgICAgPEN1c3RvbUNvbCB3aWR0aD17c2VsZWN0b3JXaWR0aH0+XHJcbiAgICAgICAgPGRpdj5cclxuICAgICAgICAgIDxTdHlsZWRTZWxlY3RcclxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBhbnkpID0+IHtcclxuICAgICAgICAgICAgICBzZXRDdXJyZW50RGF0YXNldChlKTtcclxuICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgc3R5bGU9e3ttaW5XaWR0aDogJzIwMHB4J319XHJcbiAgICAgICAgICAgIHZhbHVlPXtkYXRhc2V0c1tjdXJyZW50RGF0YXNldE5hbWVJbmRleF19XHJcbiAgICAgICAgICAgIGRyb3Bkb3duTWF0Y2hTZWxlY3RXaWR0aD17ZmFsc2V9XHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHNldFNlbGVjdCghb3BlblNlbGVjdCl9XHJcbiAgICAgICAgICAgIG9wZW49e29wZW5TZWxlY3R9XHJcbiAgICAgICAgICAgIHNob3dTZWFyY2g9e3RydWV9XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIHtyZXN1bHRzX2dyb3VwZWQubWFwKChyZXN1bHQpID0+IChcclxuICAgICAgICAgICAgICA8T3B0aW9uXHJcbiAgICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICAgIHNldFNlbGVjdChmYWxzZSk7XHJcbiAgICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICAgICAgdmFsdWU9e3Jlc3VsdC5kYXRhc2V0fVxyXG4gICAgICAgICAgICAgICAga2V5PXtyZXN1bHQuZGF0YXNldH1cclxuICAgICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICAgICAgICA8T3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgICAgIDxTcGluIC8+XHJcbiAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxyXG4gICAgICAgICAgICAgICAgKSA6IChcclxuICAgICAgICAgICAgICAgICAgPHA+e3Jlc3VsdC5kYXRhc2V0fTwvcD5cclxuICAgICAgICAgICAgICAgICl9XHJcbiAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICkpfVxyXG4gICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgPC9kaXY+XHJcbiAgICAgIDwvQ3VzdG9tQ29sPlxyXG4gICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgIDxDb2w+XHJcbiAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgZGlzYWJsZWQ9eyFkYXRhc2V0c1tjdXJyZW50RGF0YXNldE5hbWVJbmRleCArIDFdfVxyXG4gICAgICAgICAgICBpY29uPXs8Q2FyZXRSaWdodEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgIHNldEN1cnJlbnREYXRhc2V0KGRhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICB9fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbD5cclxuICAgICAgKX1cclxuICAgIDwvUm93PlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
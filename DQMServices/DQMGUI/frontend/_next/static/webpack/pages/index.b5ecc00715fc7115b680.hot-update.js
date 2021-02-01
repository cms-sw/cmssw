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
        lineNumber: 77,
        columnNumber: 15
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_3__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 85,
        columnNumber: 19
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 86,
        columnNumber: 21
      }
    })) : __jsx("p", {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 89,
        columnNumber: 21
      }
    }, result.dataset));
  })))), !withoutArrows && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    disabled: !datasets[currentDatasetNameIndex + 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 101,
        columnNumber: 19
      }
    }),
    onClick: function onClick() {
      setCurrentDataset(datasets[currentDatasetNameIndex + 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXRzQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiRGF0YXNldHNCcm93c2VyIiwid2l0aG91dEFycm93cyIsInNldEN1cnJlbnREYXRhc2V0Iiwic2VsZWN0b3JXaWR0aCIsInF1ZXJ5IiwiY3VycmVudF9kYXRhc2V0X25hbWUiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJ1c2VTdGF0ZSIsIm9wZW5TZWxlY3QiLCJzZXRTZWxlY3QiLCJjdXJyZW50RGF0YXNldE5hbWVJbmRleCIsInNldEN1cnJlbnREYXRhc2V0TmFtZUluZGV4IiwicnVuX251bWJlciIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsImRhdGFzZXRzIiwibWFwIiwicmVzdWx0IiwiZGF0YXNldCIsInVzZUVmZmVjdCIsInF1ZXJ5X2RhdGFzZXQiLCJkYXRhc2V0X25hbWUiLCJpbmRleE9mIiwiZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBRUE7QUFJQTtBQUVBO0lBV1FBLE0sR0FBV0MsMkMsQ0FBWEQsTTtBQUVELElBQU1FLGVBQWUsR0FBRyxTQUFsQkEsZUFBa0IsT0FPSDtBQUFBOztBQUFBLE1BTjFCQyxhQU0wQixRQU4xQkEsYUFNMEI7QUFBQSxNQUwxQkMsaUJBSzBCLFFBTDFCQSxpQkFLMEI7QUFBQSxNQUoxQkMsYUFJMEIsUUFKMUJBLGFBSTBCO0FBQUEsTUFIMUJDLEtBRzBCLFFBSDFCQSxLQUcwQjtBQUFBLE1BRjFCQyxvQkFFMEIsUUFGMUJBLG9CQUUwQjtBQUFBLE1BRDFCQyxrQkFDMEIsUUFEMUJBLGtCQUMwQjs7QUFBQSxrQkFDTUMsc0RBQVEsQ0FBQyxLQUFELENBRGQ7QUFBQSxNQUNuQkMsVUFEbUI7QUFBQSxNQUNQQyxTQURPOztBQUFBLG1CQUVvQ0Ys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QVgsQ0FaTixDQURtQjtBQUFBLEdBQXBCLENBVkgsQ0FERixDQURGLENBYkYsRUE2Q0csQ0FBQ2xCLGFBQUQsSUFDQyxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsUUFBSSxFQUFDLE1BRFA7QUFFRSxZQUFRLEVBQUUsQ0FBQ2UsUUFBUSxDQUFDTix1QkFBdUIsR0FBRyxDQUEzQixDQUZyQjtBQUdFLFFBQUksRUFBRSxNQUFDLGtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFIUjtBQUlFLFdBQU8sRUFBRSxtQkFBTTtBQUNiUix1QkFBaUIsQ0FBQ2MsUUFBUSxDQUFDTix1QkFBdUIsR0FBRyxDQUEzQixDQUFULENBQWpCO0FBQ0QsS0FOSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0E5Q0osQ0FERjtBQTRERCxDQXRGTTs7R0FBTVYsZTtVQWE0QmEsMEQ7OztLQWI1QmIsZSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5iNWVjYzAwNzE1ZmM3MTE1YjY4MC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IENvbCwgU2VsZWN0LCBSb3csIFNwaW4sIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBDYXJldFJpZ2h0RmlsbGVkLCBDYXJldExlZnRGaWxsZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQge1xyXG4gIFN0eWxlZFNlbGVjdCxcclxuICBPcHRpb25QYXJhZ3JhcGgsXHJcbn0gZnJvbSAnLi4vLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VTZWFyY2gnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBDdXN0b21Db2wgfSBmcm9tICcuLi8uLi9zdHlsZWRDb21wb25lbnRzJztcclxuXHJcbmludGVyZmFjZSBEYXRhc2V0c0Jyb3dzZXJQcm9wcyB7XHJcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XHJcbiAgc2V0Q3VycmVudERhdGFzZXQoY3VycmVudERhdGFzZXQ6IHN0cmluZyk6IHZvaWQ7XHJcbiAgd2l0aG91dEFycm93cz86IGJvb2xlYW47XHJcbiAgc2VsZWN0b3JXaWR0aD86IHN0cmluZztcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZT86IHN0cmluZztcclxuICBjdXJyZW50X3J1bl9udW1iZXI/OiBzdHJpbmc7XHJcbn1cclxuXHJcbmNvbnN0IHsgT3B0aW9uIH0gPSBTZWxlY3Q7XHJcblxyXG5leHBvcnQgY29uc3QgRGF0YXNldHNCcm93c2VyID0gKHtcclxuICB3aXRob3V0QXJyb3dzLFxyXG4gIHNldEN1cnJlbnREYXRhc2V0LFxyXG4gIHNlbGVjdG9yV2lkdGgsXHJcbiAgcXVlcnksXHJcbiAgY3VycmVudF9kYXRhc2V0X25hbWUsXHJcbiAgY3VycmVudF9ydW5fbnVtYmVyLFxyXG59OiBEYXRhc2V0c0Jyb3dzZXJQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvcGVuU2VsZWN0LCBzZXRTZWxlY3RdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IFtjdXJyZW50RGF0YXNldE5hbWVJbmRleCwgc2V0Q3VycmVudERhdGFzZXROYW1lSW5kZXhdID0gdXNlU3RhdGU8XHJcbiAgICBudW1iZXJcclxuICA+KDApO1xyXG4gIGNvbnN0IHJ1bl9udW1iZXIgPSBjdXJyZW50X3J1bl9udW1iZXIgPyBjdXJyZW50X3J1bl9udW1iZXIgOiBxdWVyeS5ydW5fbnVtYmVyO1xyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBpc0xvYWRpbmcgfSA9IHVzZVNlYXJjaChydW5fbnVtYmVyLCAnJyk7XHJcblxyXG4gIGNvbnN0IGRhdGFzZXRzID0gcmVzdWx0c19ncm91cGVkLm1hcCgocmVzdWx0KSA9PiB7XHJcbiAgICByZXR1cm4gcmVzdWx0LmRhdGFzZXQ7XHJcbiAgfSk7XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBxdWVyeV9kYXRhc2V0ID0gY3VycmVudF9kYXRhc2V0X25hbWVcclxuICAgICAgPyBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgICA6IHF1ZXJ5LmRhdGFzZXRfbmFtZTtcclxuICAgIHNldEN1cnJlbnREYXRhc2V0TmFtZUluZGV4KGRhdGFzZXRzLmluZGV4T2YocXVlcnlfZGF0YXNldCkpO1xyXG4gIH0sIFtpc0xvYWRpbmcsIGRhdGFzZXRzXSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Um93IGp1c3RpZnk9XCJjZW50ZXJcIiBhbGlnbj1cIm1pZGRsZVwiPlxyXG4gICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgIDxDb2w+XHJcbiAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgIGRpc2FibGVkPXshZGF0YXNldHNbY3VycmVudERhdGFzZXROYW1lSW5kZXggLSAxXX1cclxuICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICBpY29uPXs8Q2FyZXRMZWZ0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgc2V0Q3VycmVudERhdGFzZXQoZGF0YXNldHNbY3VycmVudERhdGFzZXROYW1lSW5kZXggLSAxXSk7XHJcbiAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sPlxyXG4gICAgICApfVxyXG4gICAgICA8Q3VzdG9tQ29sIHdpZHRoPXtzZWxlY3RvcldpZHRofT5cclxuICAgICAgICA8ZGl2PlxyXG4gICAgICAgICAgPFN0eWxlZFNlbGVjdFxyXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHNldEN1cnJlbnREYXRhc2V0KGUpO1xyXG4gICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICB2YWx1ZT17ZGF0YXNldHNbY3VycmVudERhdGFzZXROYW1lSW5kZXhdfVxyXG4gICAgICAgICAgICBkcm9wZG93bk1hdGNoU2VsZWN0V2lkdGg9e2ZhbHNlfVxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpfVxyXG4gICAgICAgICAgICBvcGVuPXtvcGVuU2VsZWN0fVxyXG4gICAgICAgICAgICBzaG93U2VhcmNoPXt0cnVlfVxyXG4gICAgICAgICAgPlxyXG4gICAgICAgICAgICB7cmVzdWx0c19ncm91cGVkLm1hcCgocmVzdWx0KSA9PiAoXHJcbiAgICAgICAgICAgICAgPE9wdGlvblxyXG4gICAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgICBzZXRTZWxlY3QoZmFsc2UpO1xyXG4gICAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgICAgIHZhbHVlPXtyZXN1bHQuZGF0YXNldH1cclxuICAgICAgICAgICAgICAgIGtleT17cmVzdWx0LmRhdGFzZXR9XHJcbiAgICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgICAge2lzTG9hZGluZyA/IChcclxuICAgICAgICAgICAgICAgICAgPE9wdGlvblBhcmFncmFwaD5cclxuICAgICAgICAgICAgICAgICAgICA8U3BpbiAvPlxyXG4gICAgICAgICAgICAgICAgICA8L09wdGlvblBhcmFncmFwaD5cclxuICAgICAgICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgICAgICAgICAgPHA+e3Jlc3VsdC5kYXRhc2V0fTwvcD5cclxuICAgICAgICAgICAgICAgICAgKX1cclxuICAgICAgICAgICAgICA8L09wdGlvbj5cclxuICAgICAgICAgICAgKSl9XHJcbiAgICAgICAgICA8L1N0eWxlZFNlbGVjdD5cclxuICAgICAgICA8L2Rpdj5cclxuICAgICAgPC9DdXN0b21Db2w+XHJcbiAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgPENvbD5cclxuICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICBkaXNhYmxlZD17IWRhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4ICsgMV19XHJcbiAgICAgICAgICAgIGljb249ezxDYXJldFJpZ2h0RmlsbGVkIC8+fVxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgc2V0Q3VycmVudERhdGFzZXQoZGF0YXNldHNbY3VycmVudERhdGFzZXROYW1lSW5kZXggKyAxXSk7XHJcbiAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sPlxyXG4gICAgICApfVxyXG4gICAgPC9Sb3c+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
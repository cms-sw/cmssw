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
    minWidth: "200px",
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
        columnNumber: 21
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXRzQnJvd3Nlci50c3giXSwibmFtZXMiOlsiT3B0aW9uIiwiU2VsZWN0IiwiRGF0YXNldHNCcm93c2VyIiwid2l0aG91dEFycm93cyIsInNldEN1cnJlbnREYXRhc2V0Iiwic2VsZWN0b3JXaWR0aCIsInF1ZXJ5IiwiY3VycmVudF9kYXRhc2V0X25hbWUiLCJjdXJyZW50X3J1bl9udW1iZXIiLCJ1c2VTdGF0ZSIsIm9wZW5TZWxlY3QiLCJzZXRTZWxlY3QiLCJjdXJyZW50RGF0YXNldE5hbWVJbmRleCIsInNldEN1cnJlbnREYXRhc2V0TmFtZUluZGV4IiwicnVuX251bWJlciIsInVzZVNlYXJjaCIsInJlc3VsdHNfZ3JvdXBlZCIsImlzTG9hZGluZyIsImRhdGFzZXRzIiwibWFwIiwicmVzdWx0IiwiZGF0YXNldCIsInVzZUVmZmVjdCIsInF1ZXJ5X2RhdGFzZXQiLCJkYXRhc2V0X25hbWUiLCJpbmRleE9mIiwiZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBRUE7QUFJQTtBQUVBO0lBV1FBLE0sR0FBV0MsMkMsQ0FBWEQsTTtBQUVELElBQU1FLGVBQWUsR0FBRyxTQUFsQkEsZUFBa0IsT0FPSDtBQUFBOztBQUFBLE1BTjFCQyxhQU0wQixRQU4xQkEsYUFNMEI7QUFBQSxNQUwxQkMsaUJBSzBCLFFBTDFCQSxpQkFLMEI7QUFBQSxNQUoxQkMsYUFJMEIsUUFKMUJBLGFBSTBCO0FBQUEsTUFIMUJDLEtBRzBCLFFBSDFCQSxLQUcwQjtBQUFBLE1BRjFCQyxvQkFFMEIsUUFGMUJBLG9CQUUwQjtBQUFBLE1BRDFCQyxrQkFDMEIsUUFEMUJBLGtCQUMwQjs7QUFBQSxrQkFDTUMsc0RBQVEsQ0FBQyxLQUFELENBRGQ7QUFBQSxNQUNuQkMsVUFEbUI7QUFBQSxNQUNQQyxTQURPOztBQUFBLG1CQUVvQ0Ys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LE9BQUlHLE1BQU0sQ0FBQ0MsT0FBWCxDQVpOLENBRG1CO0FBQUEsR0FBcEIsQ0FYSCxDQURGLENBREYsQ0FiRixFQThDRyxDQUFDbEIsYUFBRCxJQUNDLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFDRSxRQUFJLEVBQUMsTUFEUDtBQUVFLFlBQVEsRUFBRSxDQUFDZSxRQUFRLENBQUNOLHVCQUF1QixHQUFHLENBQTNCLENBRnJCO0FBR0UsUUFBSSxFQUFFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUhSO0FBSUUsV0FBTyxFQUFFLG1CQUFNO0FBQ2JSLHVCQUFpQixDQUFDYyxRQUFRLENBQUNOLHVCQUF1QixHQUFHLENBQTNCLENBQVQsQ0FBakI7QUFDRCxLQU5IO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQS9DSixDQURGO0FBNkRELENBdkZNOztHQUFNVixlO1VBYTRCYSwwRDs7O0tBYjVCYixlIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjhkMmRiNWI5YmNmMGYwODc0ZjZjLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgQ29sLCBTZWxlY3QsIFJvdywgU3BpbiwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IENhcmV0UmlnaHRGaWxsZWQsIENhcmV0TGVmdEZpbGxlZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkU2VsZWN0LFxyXG4gIE9wdGlvblBhcmFncmFwaCxcclxufSBmcm9tICcuLi8uLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IEN1c3RvbUNvbCB9IGZyb20gJy4uLy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5cclxuaW50ZXJmYWNlIERhdGFzZXRzQnJvd3NlclByb3BzIHtcclxuICBxdWVyeTogUXVlcnlQcm9wcztcclxuICBzZXRDdXJyZW50RGF0YXNldChjdXJyZW50RGF0YXNldDogc3RyaW5nKTogdm9pZDtcclxuICB3aXRob3V0QXJyb3dzPzogYm9vbGVhbjtcclxuICBzZWxlY3RvcldpZHRoPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xyXG4gIGN1cnJlbnRfcnVuX251bWJlcj86IHN0cmluZztcclxufVxyXG5cclxuY29uc3QgeyBPcHRpb24gfSA9IFNlbGVjdDtcclxuXHJcbmV4cG9ydCBjb25zdCBEYXRhc2V0c0Jyb3dzZXIgPSAoe1xyXG4gIHdpdGhvdXRBcnJvd3MsXHJcbiAgc2V0Q3VycmVudERhdGFzZXQsXHJcbiAgc2VsZWN0b3JXaWR0aCxcclxuICBxdWVyeSxcclxuICBjdXJyZW50X2RhdGFzZXRfbmFtZSxcclxuICBjdXJyZW50X3J1bl9udW1iZXIsXHJcbn06IERhdGFzZXRzQnJvd3NlclByb3BzKSA9PiB7XHJcbiAgY29uc3QgW29wZW5TZWxlY3QsIHNldFNlbGVjdF0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcbiAgY29uc3QgW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4LCBzZXRDdXJyZW50RGF0YXNldE5hbWVJbmRleF0gPSB1c2VTdGF0ZTxcclxuICAgIG51bWJlclxyXG4gID4oMCk7XHJcbiAgY29uc3QgcnVuX251bWJlciA9IGN1cnJlbnRfcnVuX251bWJlciA/IGN1cnJlbnRfcnVuX251bWJlciA6IHF1ZXJ5LnJ1bl9udW1iZXI7XHJcbiAgY29uc3QgeyByZXN1bHRzX2dyb3VwZWQsIGlzTG9hZGluZyB9ID0gdXNlU2VhcmNoKHJ1bl9udW1iZXIsICcnKTtcclxuXHJcbiAgY29uc3QgZGF0YXNldHMgPSByZXN1bHRzX2dyb3VwZWQubWFwKChyZXN1bHQpID0+IHtcclxuICAgIHJldHVybiByZXN1bHQuZGF0YXNldDtcclxuICB9KTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHF1ZXJ5X2RhdGFzZXQgPSBjdXJyZW50X2RhdGFzZXRfbmFtZVxyXG4gICAgICA/IGN1cnJlbnRfZGF0YXNldF9uYW1lXHJcbiAgICAgIDogcXVlcnkuZGF0YXNldF9uYW1lO1xyXG4gICAgc2V0Q3VycmVudERhdGFzZXROYW1lSW5kZXgoZGF0YXNldHMuaW5kZXhPZihxdWVyeV9kYXRhc2V0KSk7XHJcbiAgfSwgW2lzTG9hZGluZywgZGF0YXNldHNdKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XHJcbiAgICAgIHshd2l0aG91dEFycm93cyAmJiAoXHJcbiAgICAgICAgPENvbD5cclxuICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgZGlzYWJsZWQ9eyFkYXRhc2V0c1tjdXJyZW50RGF0YXNldE5hbWVJbmRleCAtIDFdfVxyXG4gICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgIGljb249ezxDYXJldExlZnRGaWxsZWQgLz59XHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICBzZXRDdXJyZW50RGF0YXNldChkYXRhc2V0c1tjdXJyZW50RGF0YXNldE5hbWVJbmRleCAtIDFdKTtcclxuICAgICAgICAgICAgfX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICl9XHJcbiAgICAgIDxDdXN0b21Db2wgd2lkdGg9e3NlbGVjdG9yV2lkdGh9PlxyXG4gICAgICAgIDxkaXY+XHJcbiAgICAgICAgICA8U3R5bGVkU2VsZWN0XHJcbiAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgc2V0Q3VycmVudERhdGFzZXQoZSk7XHJcbiAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgIG1pbldpZHRoPScyMDBweCdcclxuICAgICAgICAgICAgdmFsdWU9e2RhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4XX1cclxuICAgICAgICAgICAgZHJvcGRvd25NYXRjaFNlbGVjdFdpZHRoPXtmYWxzZX1cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KCFvcGVuU2VsZWN0KX1cclxuICAgICAgICAgICAgb3Blbj17b3BlblNlbGVjdH1cclxuICAgICAgICAgICAgc2hvd1NlYXJjaD17dHJ1ZX1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAge3Jlc3VsdHNfZ3JvdXBlZC5tYXAoKHJlc3VsdCkgPT4gKFxyXG4gICAgICAgICAgICAgIDxPcHRpb25cclxuICAgICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgICAgc2V0U2VsZWN0KGZhbHNlKTtcclxuICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgICB2YWx1ZT17cmVzdWx0LmRhdGFzZXR9XHJcbiAgICAgICAgICAgICAgICBrZXk9e3Jlc3VsdC5kYXRhc2V0fVxyXG4gICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgIHtpc0xvYWRpbmcgPyAoXHJcbiAgICAgICAgICAgICAgICAgIDxPcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICAgICAgPFNwaW4gLz5cclxuICAgICAgICAgICAgICAgICAgPC9PcHRpb25QYXJhZ3JhcGg+XHJcbiAgICAgICAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICAgICAgICAgIDxwPntyZXN1bHQuZGF0YXNldH08L3A+XHJcbiAgICAgICAgICAgICAgICAgICl9XHJcbiAgICAgICAgICAgICAgPC9PcHRpb24+XHJcbiAgICAgICAgICAgICkpfVxyXG4gICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XHJcbiAgICAgICAgPC9kaXY+XHJcbiAgICAgIDwvQ3VzdG9tQ29sPlxyXG4gICAgICB7IXdpdGhvdXRBcnJvd3MgJiYgKFxyXG4gICAgICAgIDxDb2w+XHJcbiAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgZGlzYWJsZWQ9eyFkYXRhc2V0c1tjdXJyZW50RGF0YXNldE5hbWVJbmRleCArIDFdfVxyXG4gICAgICAgICAgICBpY29uPXs8Q2FyZXRSaWdodEZpbGxlZCAvPn1cclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgIHNldEN1cnJlbnREYXRhc2V0KGRhdGFzZXRzW2N1cnJlbnREYXRhc2V0TmFtZUluZGV4ICsgMV0pO1xyXG4gICAgICAgICAgICB9fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbD5cclxuICAgICAgKX1cclxuICAgIDwvUm93PlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
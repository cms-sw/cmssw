webpackHotUpdate_N_E("pages/index",{

/***/ "./components/browsing/index.tsx":
/*!***************************************!*\
  !*** ./components/browsing/index.tsx ***!
  \***************************************/
/*! exports provided: Browser */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Browser", function() { return Browser; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _datasetsBrowsing_datasetsBrowser__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./datasetsBrowsing/datasetsBrowser */ "./components/browsing/datasetsBrowsing/datasetsBrowser.tsx");
/* harmony import */ var _datasetsBrowsing_datasetNameBuilder__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./datasetsBrowsing/datasetNameBuilder */ "./components/browsing/datasetsBrowsing/datasetNameBuilder.tsx");
/* harmony import */ var _runsBrowser__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./runsBrowser */ "./components/browsing/runsBrowser.tsx");
/* harmony import */ var _lumesectionBroweser__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./lumesectionBroweser */ "./components/browsing/lumesectionBroweser.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _menu__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../menu */ "./components/menu.tsx");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_11___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_11__);
/* harmony import */ var _hooks_useChangeRouter__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../hooks/useChangeRouter */ "./hooks/useChangeRouter.tsx");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/browsing/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;















var Browser = function Browser() {
  _s();

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(_constants__WEBPACK_IMPORTED_MODULE_8__["dataSetSelections"][0].value),
      datasetOption = _useState[0],
      setDatasetOption = _useState[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_11__["useRouter"])();
  var query = router.query;
  var run_number = query.run_number ? query.run_number : '';
  var dataset_name = query.dataset_name ? query.dataset_name : '';
  var lumi = query.lumi ? parseInt(query.lumi) : NaN;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_0___default.a.useContext(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__["store"]),
      setLumisection = _React$useContext.setLumisection;

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(run_number),
      currentRunNumber = _useState2[0],
      setCurrentRunNumber = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(dataset_name),
      currentDataset = _useState3[0],
      setCurrentDataset = _useState3[1];

  var lumisectionsChangeHandler = function lumisectionsChangeHandler(lumi) {
    //in main navigation when lumisection is changed, new value have to be set to url
    Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_14__["changeRouter"])(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_14__["getChangedQueryParams"])({
      lumi: lumi
    }, query)); //setLumisection from store(using useContext) set lumisection value globally.
    //This set value is reachable for lumisection browser in free search dialog (you can see it, when search button next to browsers is clicked).
    //Both lumisection browser have different handlers, they have to act differently according to their place:
    //IN THE MAIN NAV: lumisection browser value in the main navigation is changed, this HAVE to be set to url;
    //FREE SEARCH DIALOG: lumisection browser value in free search dialog is changed it HASN'T to be set to url immediately, just when button 'ok'
    //in dialog is clicked THEN value is set to url. So, useContext let us to change lumi value globally without changing url, when wee no need that.
    //And in this handler lumi value set to useContext store is used as initial lumi value in free search dialog.

    setLumisection(lumi);
  };

  if (currentRunNumber !== query.run_number || currentDataset !== query.dataset_name) {
    Object(_hooks_useChangeRouter__WEBPACK_IMPORTED_MODULE_12__["useChangeRouter"])({
      run_number: currentRunNumber,
      dataset_name: currentDataset
    }, [currentRunNumber, currentDataset], true);
  } //make changes through context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 64,
      columnNumber: 5
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 65,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 66,
      columnNumber: 9
    }
  }, __jsx(_runsBrowser__WEBPACK_IMPORTED_MODULE_6__["RunBrowser"], {
    query: query,
    setCurrentRunNumber: setCurrentRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 67,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 9
    }
  }, _config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].new_back_end.lumisections_on && __jsx(_lumesectionBroweser__WEBPACK_IMPORTED_MODULE_7__["LumesectionBrowser"], {
    currentLumisection: lumi,
    currentRunNumber: currentRunNumber,
    currentDataset: currentDataset,
    handler: lumisectionsChangeHandler,
    color: "white",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 71,
      columnNumber: 13
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: __jsx(_menu__WEBPACK_IMPORTED_MODULE_10__["DropdownMenu"], {
      options: _constants__WEBPACK_IMPORTED_MODULE_8__["dataSetSelections"],
      action: setDatasetOption,
      defaultValue: _constants__WEBPACK_IMPORTED_MODULE_8__["dataSetSelections"][0],
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 83,
        columnNumber: 13
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 80,
      columnNumber: 9
    }
  }, datasetOption === _constants__WEBPACK_IMPORTED_MODULE_8__["dataSetSelections"][0].value ? __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 91,
      columnNumber: 13
    }
  }, __jsx(_datasetsBrowsing_datasetsBrowser__WEBPACK_IMPORTED_MODULE_4__["DatasetsBrowser"], {
    setCurrentDataset: setCurrentDataset,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 92,
      columnNumber: 15
    }
  })) : __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 15
    }
  }, __jsx(_datasetsBrowsing_datasetNameBuilder__WEBPACK_IMPORTED_MODULE_5__["DatasetsBuilder"], {
    currentRunNumber: currentRunNumber,
    currentDataset: currentDataset,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 17
    }
  })))));
};

_s(Browser, "KTYDnSiViPjtY8ABnrRMjWtO4eU=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_11__["useRouter"], _hooks_useChangeRouter__WEBPACK_IMPORTED_MODULE_12__["useChangeRouter"]];
});

_c = Browser;

var _c;

$RefreshReg$(_c, "Browser");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9pbmRleC50c3giXSwibmFtZXMiOlsiQnJvd3NlciIsInVzZVN0YXRlIiwiZGF0YVNldFNlbGVjdGlvbnMiLCJ2YWx1ZSIsImRhdGFzZXRPcHRpb24iLCJzZXREYXRhc2V0T3B0aW9uIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwibHVtaSIsInBhcnNlSW50IiwiTmFOIiwiUmVhY3QiLCJ1c2VDb250ZXh0Iiwic3RvcmUiLCJzZXRMdW1pc2VjdGlvbiIsImN1cnJlbnRSdW5OdW1iZXIiLCJzZXRDdXJyZW50UnVuTnVtYmVyIiwiY3VycmVudERhdGFzZXQiLCJzZXRDdXJyZW50RGF0YXNldCIsImx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIiLCJjaGFuZ2VSb3V0ZXIiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJ1c2VDaGFuZ2VSb3V0ZXIiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUtPLElBQU1BLE9BQU8sR0FBRyxTQUFWQSxPQUFVLEdBQU07QUFBQTs7QUFBQSxrQkFDZUMsc0RBQVEsQ0FDaERDLDREQUFpQixDQUFDLENBQUQsQ0FBakIsQ0FBcUJDLEtBRDJCLENBRHZCO0FBQUEsTUFDcEJDLGFBRG9CO0FBQUEsTUFDTEMsZ0JBREs7O0FBSTNCLE1BQU1DLE1BQU0sR0FBR0MsOERBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDO0FBRUEsTUFBTUMsVUFBVSxHQUFHRCxLQUFLLENBQUNDLFVBQU4sR0FBbUJELEtB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O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREQsR0FRRyxNQUFDLCtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLG9GQUFEO0FBQ0Usb0JBQWdCLEVBQUVVLGdCQURwQjtBQUVFLGtCQUFjLEVBQUVFLGNBRmxCO0FBR0UsU0FBSyxFQUFFWixLQUhUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQWxCTixDQWZGLENBREYsQ0FERjtBQStDRCxDQXhGTTs7R0FBTVIsTztVQUlJTyxzRCxFQTBCYmtCLHVFOzs7S0E5QlN6QixPIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjY0ODc4NzE5ODdhNDZiNmJmMTQzLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBXcmFwcGVyRGl2IH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRGF0YXNldHNCcm93c2VyIH0gZnJvbSAnLi9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXRzQnJvd3Nlcic7XG5pbXBvcnQgeyBEYXRhc2V0c0J1aWxkZXIgfSBmcm9tICcuL2RhdGFzZXRzQnJvd3NpbmcvZGF0YXNldE5hbWVCdWlsZGVyJztcbmltcG9ydCB7IFJ1bkJyb3dzZXIgfSBmcm9tICcuL3J1bnNCcm93c2VyJztcbmltcG9ydCB7IEx1bWVzZWN0aW9uQnJvd3NlciB9IGZyb20gJy4vbHVtZXNlY3Rpb25Ccm93ZXNlcic7XG5pbXBvcnQgeyBkYXRhU2V0U2VsZWN0aW9ucyB9IGZyb20gJy4uL2NvbnN0YW50cyc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRHJvcGRvd25NZW51IH0gZnJvbSAnLi4vbWVudSc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdXNlQ2hhbmdlUm91dGVyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlQ2hhbmdlUm91dGVyJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7XG4gIGNoYW5nZVJvdXRlcixcbiAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLFxufSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgQnJvd3NlciA9ICgpID0+IHtcbiAgY29uc3QgW2RhdGFzZXRPcHRpb24sIHNldERhdGFzZXRPcHRpb25dID0gdXNlU3RhdGUoXG4gICAgZGF0YVNldFNlbGVjdGlvbnNbMF0udmFsdWVcbiAgKTtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IHJ1bl9udW1iZXIgPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xuICBjb25zdCBkYXRhc2V0X25hbWUgPSBxdWVyeS5kYXRhc2V0X25hbWUgPyBxdWVyeS5kYXRhc2V0X25hbWUgOiAnJztcbiAgY29uc3QgbHVtaSA9IHF1ZXJ5Lmx1bWkgPyBwYXJzZUludChxdWVyeS5sdW1pKSA6IE5hTjtcblxuICBjb25zdCB7IHNldEx1bWlzZWN0aW9uIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXIsIHNldEN1cnJlbnRSdW5OdW1iZXJdID0gdXNlU3RhdGUocnVuX251bWJlcik7XG4gIGNvbnN0IFtjdXJyZW50RGF0YXNldCwgc2V0Q3VycmVudERhdGFzZXRdID0gdXNlU3RhdGU8c3RyaW5nPihkYXRhc2V0X25hbWUpO1xuXG4gIGNvbnN0IGx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIgPSAobHVtaTogbnVtYmVyKSA9PiB7XG4gICAgLy9pbiBtYWluIG5hdmlnYXRpb24gd2hlbiBsdW1pc2VjdGlvbiBpcyBjaGFuZ2VkLCBuZXcgdmFsdWUgaGF2ZSB0byBiZSBzZXQgdG8gdXJsXG4gICAgY2hhbmdlUm91dGVyKGdldENoYW5nZWRRdWVyeVBhcmFtcyh7IGx1bWk6IGx1bWkgfSwgcXVlcnkpKTtcbiAgICAvL3NldEx1bWlzZWN0aW9uIGZyb20gc3RvcmUodXNpbmcgdXNlQ29udGV4dCkgc2V0IGx1bWlzZWN0aW9uIHZhbHVlIGdsb2JhbGx5LlxuICAgIC8vVGhpcyBzZXQgdmFsdWUgaXMgcmVhY2hhYmxlIGZvciBsdW1pc2VjdGlvbiBicm93c2VyIGluIGZyZWUgc2VhcmNoIGRpYWxvZyAoeW91IGNhbiBzZWUgaXQsIHdoZW4gc2VhcmNoIGJ1dHRvbiBuZXh0IHRvIGJyb3dzZXJzIGlzIGNsaWNrZWQpLlxuXG4gICAgLy9Cb3RoIGx1bWlzZWN0aW9uIGJyb3dzZXIgaGF2ZSBkaWZmZXJlbnQgaGFuZGxlcnMsIHRoZXkgaGF2ZSB0byBhY3QgZGlmZmVyZW50bHkgYWNjb3JkaW5nIHRvIHRoZWlyIHBsYWNlOlxuICAgIC8vSU4gVEhFIE1BSU4gTkFWOiBsdW1pc2VjdGlvbiBicm93c2VyIHZhbHVlIGluIHRoZSBtYWluIG5hdmlnYXRpb24gaXMgY2hhbmdlZCwgdGhpcyBIQVZFIHRvIGJlIHNldCB0byB1cmw7XG4gICAgLy9GUkVFIFNFQVJDSCBESUFMT0c6IGx1bWlzZWN0aW9uIGJyb3dzZXIgdmFsdWUgaW4gZnJlZSBzZWFyY2ggZGlhbG9nIGlzIGNoYW5nZWQgaXQgSEFTTidUIHRvIGJlIHNldCB0byB1cmwgaW1tZWRpYXRlbHksIGp1c3Qgd2hlbiBidXR0b24gJ29rJ1xuICAgIC8vaW4gZGlhbG9nIGlzIGNsaWNrZWQgVEhFTiB2YWx1ZSBpcyBzZXQgdG8gdXJsLiBTbywgdXNlQ29udGV4dCBsZXQgdXMgdG8gY2hhbmdlIGx1bWkgdmFsdWUgZ2xvYmFsbHkgd2l0aG91dCBjaGFuZ2luZyB1cmwsIHdoZW4gd2VlIG5vIG5lZWQgdGhhdC5cbiAgICAvL0FuZCBpbiB0aGlzIGhhbmRsZXIgbHVtaSB2YWx1ZSBzZXQgdG8gdXNlQ29udGV4dCBzdG9yZSBpcyB1c2VkIGFzIGluaXRpYWwgbHVtaSB2YWx1ZSBpbiBmcmVlIHNlYXJjaCBkaWFsb2cuXG4gICAgc2V0THVtaXNlY3Rpb24obHVtaSk7XG4gIH07XG5cbiAgaWYgKGN1cnJlbnRSdW5OdW1iZXIgIT09IHF1ZXJ5LnJ1bl9udW1iZXIgfHwgY3VycmVudERhdGFzZXQgIT09IHF1ZXJ5LmRhdGFzZXRfbmFtZSkge1xuICAgIHVzZUNoYW5nZVJvdXRlcihcbiAgICAgIHtcbiAgICAgICAgcnVuX251bWJlcjogY3VycmVudFJ1bk51bWJlcixcbiAgICAgICAgZGF0YXNldF9uYW1lOiBjdXJyZW50RGF0YXNldCxcbiAgICAgIH0sXG4gICAgICBbY3VycmVudFJ1bk51bWJlciwgY3VycmVudERhdGFzZXRdLFxuICAgICAgdHJ1ZVxuICAgICk7XG4gIH1cblxuICAvL21ha2UgY2hhbmdlcyB0aHJvdWdoIGNvbnRleHRcbiAgcmV0dXJuIChcbiAgICA8Rm9ybT5cbiAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICA8V3JhcHBlckRpdj5cbiAgICAgICAgICA8UnVuQnJvd3NlciBxdWVyeT17cXVlcnl9IHNldEN1cnJlbnRSdW5OdW1iZXI9e3NldEN1cnJlbnRSdW5OdW1iZXJ9IC8+XG4gICAgICAgIDwvV3JhcHBlckRpdj5cbiAgICAgICAgPFdyYXBwZXJEaXY+XG4gICAgICAgICAge2Z1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLmx1bWlzZWN0aW9uc19vbiAmJiAoXG4gICAgICAgICAgICA8THVtZXNlY3Rpb25Ccm93c2VyXG4gICAgICAgICAgICAgIGN1cnJlbnRMdW1pc2VjdGlvbj17bHVtaX1cbiAgICAgICAgICAgICAgY3VycmVudFJ1bk51bWJlcj17Y3VycmVudFJ1bk51bWJlcn1cbiAgICAgICAgICAgICAgY3VycmVudERhdGFzZXQ9e2N1cnJlbnREYXRhc2V0fVxuICAgICAgICAgICAgICBoYW5kbGVyPXtsdW1pc2VjdGlvbnNDaGFuZ2VIYW5kbGVyfVxuICAgICAgICAgICAgICBjb2xvcj1cIndoaXRlXCJcbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgKX1cbiAgICAgICAgPC9XcmFwcGVyRGl2PlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW1cbiAgICAgICAgICBsYWJlbGNvbG9yPVwid2hpdGVcIlxuICAgICAgICAgIGxhYmVsPXtcbiAgICAgICAgICAgIDxEcm9wZG93bk1lbnVcbiAgICAgICAgICAgICAgb3B0aW9ucz17ZGF0YVNldFNlbGVjdGlvbnN9XG4gICAgICAgICAgICAgIGFjdGlvbj17c2V0RGF0YXNldE9wdGlvbn1cbiAgICAgICAgICAgICAgZGVmYXVsdFZhbHVlPXtkYXRhU2V0U2VsZWN0aW9uc1swXX1cbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgfVxuICAgICAgICA+XG4gICAgICAgICAge2RhdGFzZXRPcHRpb24gPT09IGRhdGFTZXRTZWxlY3Rpb25zWzBdLnZhbHVlID8gKFxuICAgICAgICAgICAgPFdyYXBwZXJEaXY+XG4gICAgICAgICAgICAgIDxEYXRhc2V0c0Jyb3dzZXJcbiAgICAgICAgICAgICAgICBzZXRDdXJyZW50RGF0YXNldD17c2V0Q3VycmVudERhdGFzZXR9XG4gICAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxuICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgPC9XcmFwcGVyRGl2PlxuICAgICAgICAgICkgOiAoXG4gICAgICAgICAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICAgICAgICAgIDxEYXRhc2V0c0J1aWxkZXJcbiAgICAgICAgICAgICAgICAgIGN1cnJlbnRSdW5OdW1iZXI9e2N1cnJlbnRSdW5OdW1iZXJ9XG4gICAgICAgICAgICAgICAgICBjdXJyZW50RGF0YXNldD17Y3VycmVudERhdGFzZXR9XG4gICAgICAgICAgICAgICAgICBxdWVyeT17cXVlcnl9XG4gICAgICAgICAgICAgICAgLz5cbiAgICAgICAgICAgICAgPC9XcmFwcGVyRGl2PlxuICAgICAgICAgICAgKX1cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgIDwvV3JhcHBlckRpdj5cbiAgICA8L0Zvcm0+XG4gICk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==
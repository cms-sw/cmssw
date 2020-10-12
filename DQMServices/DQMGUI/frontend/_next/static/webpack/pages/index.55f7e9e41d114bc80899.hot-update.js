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
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
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

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_0___default.a.useContext(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_12__["store"]),
      setLumisection = _React$useContext.setLumisection;

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(run_number),
      currentRunNumber = _useState2[0],
      setCurrentRunNumber = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(dataset_name),
      currentDataset = _useState3[0],
      setCurrentDataset = _useState3[1];

  var lumisectionsChangeHandler = function lumisectionsChangeHandler(lumi) {
    //in main navigation when lumisection is changed, new value have to be set to url
    Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_13__["changeRouter"])(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_13__["getChangedQueryParams"])({
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
    Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_13__["changeRouter"])(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_13__["getChangedQueryParams"])({
      run_number: currentRunNumber,
      dataset_name: currentDataset
    }, query));
  } //make changes through context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 65,
      columnNumber: 5
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 66,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 67,
      columnNumber: 9
    }
  }, __jsx(_runsBrowser__WEBPACK_IMPORTED_MODULE_6__["RunBrowser"], {
    query: query,
    setCurrentRunNumber: setCurrentRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
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
      lineNumber: 72,
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
        lineNumber: 84,
        columnNumber: 13
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 9
    }
  }, datasetOption === _constants__WEBPACK_IMPORTED_MODULE_8__["dataSetSelections"][0].value ? __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 92,
      columnNumber: 13
    }
  }, __jsx(_datasetsBrowsing_datasetsBrowser__WEBPACK_IMPORTED_MODULE_4__["DatasetsBrowser"], {
    setCurrentDataset: setCurrentDataset,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 15
    }
  })) : __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_3__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 15
    }
  }, __jsx(_datasetsBrowsing_datasetNameBuilder__WEBPACK_IMPORTED_MODULE_5__["DatasetsBuilder"], {
    currentRunNumber: currentRunNumber,
    currentDataset: currentDataset,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 17
    }
  })))));
};

_s(Browser, "M4KDYbLXo4iXQuKspBjg1J11SAI=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_11__["useRouter"]];
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9pbmRleC50c3giXSwibmFtZXMiOlsiQnJvd3NlciIsInVzZVN0YXRlIiwiZGF0YVNldFNlbGVjdGlvbnMiLCJ2YWx1ZSIsImRhdGFzZXRPcHRpb24iLCJzZXREYXRhc2V0T3B0aW9uIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwibHVtaSIsInBhcnNlSW50IiwiTmFOIiwiUmVhY3QiLCJ1c2VDb250ZXh0Iiwic3RvcmUiLCJzZXRMdW1pc2VjdGlvbiIsImN1cnJlbnRSdW5OdW1iZXIiLCJzZXRDdXJyZW50UnVuTnVtYmVyIiwiY3VycmVudERhdGFzZXQiLCJzZXRDdXJyZW50RGF0YXNldCIsImx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIiLCJjaGFuZ2VSb3V0ZXIiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBR0E7QUFDQTtBQUtPLElBQU1BLE9BQU8sR0FBRyxTQUFWQSxPQUFVLEdBQU07QUFBQTs7QUFBQSxrQkFDZUMsc0RBQVEsQ0FDaERDLDREQUFpQixDQUFDLENBQUQsQ0FBakIsQ0FBcUJDLEtBRDJCLENBRHZCO0FBQUEsTUFDcEJDLGFBRG9CO0FBQUEsTUFDTEMsZ0JBREs7O0FBSTNCLE1BQU1DLE1BQU0sR0FBR0MsOERBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDO0FBRUEsTUFBTUMsVUFBVSxHQUFHRCxLQUFLLENBQUNDLFVBQU4sR0FBbUJELEtBQUssQ0FBQ0MsVUFBekIsR0FBc0MsRUFB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QVFHLE1BQUMsK0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsb0ZBQUQ7QUFDRSxvQkFBZ0IsRUFBRVUsZ0JBRHBCO0FBRUUsa0JBQWMsRUFBRUUsY0FGbEI7QUFHRSxTQUFLLEVBQUVaLEtBSFQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBbEJOLENBZkYsQ0FERixDQURGO0FBK0NELENBekZNOztHQUFNUixPO1VBSUlPLHNEOzs7S0FKSlAsTyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC41NWY3ZTllNDFkMTE0YmM4MDg5OS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IEZvcm0gZnJvbSAnYW50ZC9saWIvZm9ybS9Gb3JtJztcblxuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgV3JhcHBlckRpdiB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IERhdGFzZXRzQnJvd3NlciB9IGZyb20gJy4vZGF0YXNldHNCcm93c2luZy9kYXRhc2V0c0Jyb3dzZXInO1xuaW1wb3J0IHsgRGF0YXNldHNCdWlsZGVyIH0gZnJvbSAnLi9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXROYW1lQnVpbGRlcic7XG5pbXBvcnQgeyBSdW5Ccm93c2VyIH0gZnJvbSAnLi9ydW5zQnJvd3Nlcic7XG5pbXBvcnQgeyBMdW1lc2VjdGlvbkJyb3dzZXIgfSBmcm9tICcuL2x1bWVzZWN0aW9uQnJvd2VzZXInO1xuaW1wb3J0IHsgZGF0YVNldFNlbGVjdGlvbnMgfSBmcm9tICcuLi9jb25zdGFudHMnO1xuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0gfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IERyb3Bkb3duTWVudSB9IGZyb20gJy4uL21lbnUnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IHVzZUNoYW5nZVJvdXRlciB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZUNoYW5nZVJvdXRlcic7XG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XG5pbXBvcnQge1xuICBjaGFuZ2VSb3V0ZXIsXG4gIGdldENoYW5nZWRRdWVyeVBhcmFtcyxcbn0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJztcblxuZXhwb3J0IGNvbnN0IEJyb3dzZXIgPSAoKSA9PiB7XG4gIGNvbnN0IFtkYXRhc2V0T3B0aW9uLCBzZXREYXRhc2V0T3B0aW9uXSA9IHVzZVN0YXRlKFxuICAgIGRhdGFTZXRTZWxlY3Rpb25zWzBdLnZhbHVlXG4gICk7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCBydW5fbnVtYmVyID0gcXVlcnkucnVuX251bWJlciA/IHF1ZXJ5LnJ1bl9udW1iZXIgOiAnJztcbiAgY29uc3QgZGF0YXNldF9uYW1lID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XG4gIGNvbnN0IGx1bWkgPSBxdWVyeS5sdW1pID8gcGFyc2VJbnQocXVlcnkubHVtaSkgOiBOYU47XG5cbiAgY29uc3QgeyBzZXRMdW1pc2VjdGlvbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XG4gIGNvbnN0IFtjdXJyZW50UnVuTnVtYmVyLCBzZXRDdXJyZW50UnVuTnVtYmVyXSA9IHVzZVN0YXRlKHJ1bl9udW1iZXIpO1xuICBjb25zdCBbY3VycmVudERhdGFzZXQsIHNldEN1cnJlbnREYXRhc2V0XSA9IHVzZVN0YXRlPHN0cmluZz4oZGF0YXNldF9uYW1lKTtcblxuICBjb25zdCBsdW1pc2VjdGlvbnNDaGFuZ2VIYW5kbGVyID0gKGx1bWk6IG51bWJlcikgPT4ge1xuICAgIC8vaW4gbWFpbiBuYXZpZ2F0aW9uIHdoZW4gbHVtaXNlY3Rpb24gaXMgY2hhbmdlZCwgbmV3IHZhbHVlIGhhdmUgdG8gYmUgc2V0IHRvIHVybFxuICAgIGNoYW5nZVJvdXRlcihnZXRDaGFuZ2VkUXVlcnlQYXJhbXMoeyBsdW1pOiBsdW1pIH0sIHF1ZXJ5KSk7XG4gICAgLy9zZXRMdW1pc2VjdGlvbiBmcm9tIHN0b3JlKHVzaW5nIHVzZUNvbnRleHQpIHNldCBsdW1pc2VjdGlvbiB2YWx1ZSBnbG9iYWxseS5cbiAgICAvL1RoaXMgc2V0IHZhbHVlIGlzIHJlYWNoYWJsZSBmb3IgbHVtaXNlY3Rpb24gYnJvd3NlciBpbiBmcmVlIHNlYXJjaCBkaWFsb2cgKHlvdSBjYW4gc2VlIGl0LCB3aGVuIHNlYXJjaCBidXR0b24gbmV4dCB0byBicm93c2VycyBpcyBjbGlja2VkKS5cblxuICAgIC8vQm90aCBsdW1pc2VjdGlvbiBicm93c2VyIGhhdmUgZGlmZmVyZW50IGhhbmRsZXJzLCB0aGV5IGhhdmUgdG8gYWN0IGRpZmZlcmVudGx5IGFjY29yZGluZyB0byB0aGVpciBwbGFjZTpcbiAgICAvL0lOIFRIRSBNQUlOIE5BVjogbHVtaXNlY3Rpb24gYnJvd3NlciB2YWx1ZSBpbiB0aGUgbWFpbiBuYXZpZ2F0aW9uIGlzIGNoYW5nZWQsIHRoaXMgSEFWRSB0byBiZSBzZXQgdG8gdXJsO1xuICAgIC8vRlJFRSBTRUFSQ0ggRElBTE9HOiBsdW1pc2VjdGlvbiBicm93c2VyIHZhbHVlIGluIGZyZWUgc2VhcmNoIGRpYWxvZyBpcyBjaGFuZ2VkIGl0IEhBU04nVCB0byBiZSBzZXQgdG8gdXJsIGltbWVkaWF0ZWx5LCBqdXN0IHdoZW4gYnV0dG9uICdvaydcbiAgICAvL2luIGRpYWxvZyBpcyBjbGlja2VkIFRIRU4gdmFsdWUgaXMgc2V0IHRvIHVybC4gU28sIHVzZUNvbnRleHQgbGV0IHVzIHRvIGNoYW5nZSBsdW1pIHZhbHVlIGdsb2JhbGx5IHdpdGhvdXQgY2hhbmdpbmcgdXJsLCB3aGVuIHdlZSBubyBuZWVkIHRoYXQuXG4gICAgLy9BbmQgaW4gdGhpcyBoYW5kbGVyIGx1bWkgdmFsdWUgc2V0IHRvIHVzZUNvbnRleHQgc3RvcmUgaXMgdXNlZCBhcyBpbml0aWFsIGx1bWkgdmFsdWUgaW4gZnJlZSBzZWFyY2ggZGlhbG9nLlxuICAgIHNldEx1bWlzZWN0aW9uKGx1bWkpO1xuICB9O1xuXG4gIGlmIChjdXJyZW50UnVuTnVtYmVyICE9PSBxdWVyeS5ydW5fbnVtYmVyIHx8IGN1cnJlbnREYXRhc2V0ICE9PSBxdWVyeS5kYXRhc2V0X25hbWUpIHtcbiAgICBjaGFuZ2VSb3V0ZXIoXG4gICAgICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMoXG4gICAgICAgIHtcbiAgICAgICAgICBydW5fbnVtYmVyOiBjdXJyZW50UnVuTnVtYmVyLFxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogY3VycmVudERhdGFzZXQsXG4gICAgICAgIH0sXG4gICAgICAgIHF1ZXJ5XG4gICAgICApXG4gICAgKTtcbiAgfVxuXG4gIC8vbWFrZSBjaGFuZ2VzIHRocm91Z2ggY29udGV4dFxuICByZXR1cm4gKFxuICAgIDxGb3JtPlxuICAgICAgPFdyYXBwZXJEaXY+XG4gICAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICAgIDxSdW5Ccm93c2VyIHF1ZXJ5PXtxdWVyeX0gc2V0Q3VycmVudFJ1bk51bWJlcj17c2V0Q3VycmVudFJ1bk51bWJlcn0gLz5cbiAgICAgICAgPC9XcmFwcGVyRGl2PlxuICAgICAgICA8V3JhcHBlckRpdj5cbiAgICAgICAgICB7ZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubHVtaXNlY3Rpb25zX29uICYmIChcbiAgICAgICAgICAgIDxMdW1lc2VjdGlvbkJyb3dzZXJcbiAgICAgICAgICAgICAgY3VycmVudEx1bWlzZWN0aW9uPXtsdW1pfVxuICAgICAgICAgICAgICBjdXJyZW50UnVuTnVtYmVyPXtjdXJyZW50UnVuTnVtYmVyfVxuICAgICAgICAgICAgICBjdXJyZW50RGF0YXNldD17Y3VycmVudERhdGFzZXR9XG4gICAgICAgICAgICAgIGhhbmRsZXI9e2x1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXJ9XG4gICAgICAgICAgICAgIGNvbG9yPVwid2hpdGVcIlxuICAgICAgICAgICAgLz5cbiAgICAgICAgICApfVxuICAgICAgICA8L1dyYXBwZXJEaXY+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbVxuICAgICAgICAgIGxhYmVsY29sb3I9XCJ3aGl0ZVwiXG4gICAgICAgICAgbGFiZWw9e1xuICAgICAgICAgICAgPERyb3Bkb3duTWVudVxuICAgICAgICAgICAgICBvcHRpb25zPXtkYXRhU2V0U2VsZWN0aW9uc31cbiAgICAgICAgICAgICAgYWN0aW9uPXtzZXREYXRhc2V0T3B0aW9ufVxuICAgICAgICAgICAgICBkZWZhdWx0VmFsdWU9e2RhdGFTZXRTZWxlY3Rpb25zWzBdfVxuICAgICAgICAgICAgLz5cbiAgICAgICAgICB9XG4gICAgICAgID5cbiAgICAgICAgICB7ZGF0YXNldE9wdGlvbiA9PT0gZGF0YVNldFNlbGVjdGlvbnNbMF0udmFsdWUgPyAoXG4gICAgICAgICAgICA8V3JhcHBlckRpdj5cbiAgICAgICAgICAgICAgPERhdGFzZXRzQnJvd3NlclxuICAgICAgICAgICAgICAgIHNldEN1cnJlbnREYXRhc2V0PXtzZXRDdXJyZW50RGF0YXNldH1cbiAgICAgICAgICAgICAgICBxdWVyeT17cXVlcnl9XG4gICAgICAgICAgICAgIC8+XG4gICAgICAgICAgICA8L1dyYXBwZXJEaXY+XG4gICAgICAgICAgKSA6IChcbiAgICAgICAgICAgICAgPFdyYXBwZXJEaXY+XG4gICAgICAgICAgICAgICAgPERhdGFzZXRzQnVpbGRlclxuICAgICAgICAgICAgICAgICAgY3VycmVudFJ1bk51bWJlcj17Y3VycmVudFJ1bk51bWJlcn1cbiAgICAgICAgICAgICAgICAgIGN1cnJlbnREYXRhc2V0PXtjdXJyZW50RGF0YXNldH1cbiAgICAgICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cbiAgICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgICA8L1dyYXBwZXJEaXY+XG4gICAgICAgICAgICApfVxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgPC9XcmFwcGVyRGl2PlxuICAgIDwvRm9ybT5cbiAgKTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9
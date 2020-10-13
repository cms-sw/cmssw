webpackHotUpdate_N_E("pages/index",{

/***/ "./components/browsing/datasetsBrowsing/datasetNameBuilder.tsx":
/*!*********************************************************************!*\
  !*** ./components/browsing/datasetsBrowsing/datasetNameBuilder.tsx ***!
  \*********************************************************************/
/*! exports provided: DatasetsBuilder */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DatasetsBuilder", function() { return DatasetsBuilder; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _partBrowser__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./partBrowser */ "./components/browsing/datasetsBrowsing/partBrowser.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _hooks_useAvailbleAndNotAvailableDatasetPartsOptions__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../hooks/useAvailbleAndNotAvailableDatasetPartsOptions */ "./hooks/useAvailbleAndNotAvailableDatasetPartsOptions.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../containers/display/utils */ "./containers/display/utils.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/browsing/datasetsBrowsing/datasetNameBuilder.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;






var DatasetsBuilder = function DatasetsBuilder(_ref) {
  _s();

  var currentDataset = _ref.currentDataset,
      query = _ref.query,
      currentRunNumber = _ref.currentRunNumber;

  var _useAvailbleAndNotAva = Object(_hooks_useAvailbleAndNotAvailableDatasetPartsOptions__WEBPACK_IMPORTED_MODULE_4__["useAvailbleAndNotAvailableDatasetPartsOptions"])(currentRunNumber, currentDataset),
      availableAndNotAvailableDatasetParts = _useAvailbleAndNotAva.availableAndNotAvailableDatasetParts,
      setSelectedParts = _useAvailbleAndNotAva.setSelectedParts,
      selectedParts = _useAvailbleAndNotAva.selectedParts,
      setLastSelectedDatasetPartValue = _useAvailbleAndNotAva.setLastSelectedDatasetPartValue,
      lastSelectedDatasetPartValue = _useAvailbleAndNotAva.lastSelectedDatasetPartValue,
      setLastSelectedDatasetPartPosition = _useAvailbleAndNotAva.setLastSelectedDatasetPartPosition,
      doesCombinationOfSelectedDatasetPartsExists = _useAvailbleAndNotAva.doesCombinationOfSelectedDatasetPartsExists,
      fullDatasetName = _useAvailbleAndNotAva.fullDatasetName;

  Object(react__WEBPACK_IMPORTED_MODULE_0__["useEffect"])(function () {
    if (doesCombinationOfSelectedDatasetPartsExists) {
      Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["changeRouter"])(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["getChangedQueryParams"])({
        dataset_name: fullDatasetName
      }, query));
    }
  }, [fullDatasetName]);
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 5
    }
  }, availableAndNotAvailableDatasetParts.map(function (part) {
    var partName = Object.keys(part)[0];
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 57,
        columnNumber: 11
      }
    }, __jsx(_partBrowser__WEBPACK_IMPORTED_MODULE_2__["PartsBrowser"], {
      restParts: part[partName].notAvailableChoices,
      part: partName,
      resultsNames: part[partName].availableChoices,
      setGroupBy: setLastSelectedDatasetPartPosition,
      setName: setLastSelectedDatasetPartValue,
      selectedName: lastSelectedDatasetPartValue //@ts-ignore
      ,
      name: selectedParts[partName],
      setSelectedParts: setSelectedParts,
      selectedParts: selectedParts,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 58,
        columnNumber: 13
      }
    }));
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 7
    }
  }, doesCombinationOfSelectedDatasetPartsExists ? __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledSuccessIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 11
    }
  }) : __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledErrorIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 77,
      columnNumber: 11
    }
  })));
};

_s(DatasetsBuilder, "Jqta3OEtovQmKnQtivpuPi6HJF8=", false, function () {
  return [_hooks_useAvailbleAndNotAvailableDatasetPartsOptions__WEBPACK_IMPORTED_MODULE_4__["useAvailbleAndNotAvailableDatasetPartsOptions"]];
});

_c = DatasetsBuilder;

var _c;

$RefreshReg$(_c, "DatasetsBuilder");

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

/***/ }),

/***/ "./components/browsing/datasetsBrowsing/partBrowser.tsx":
/*!**************************************************************!*\
  !*** ./components/browsing/datasetsBrowsing/partBrowser.tsx ***!
  \**************************************************************/
/*! exports provided: PartsBrowser */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PartsBrowser", function() { return PartsBrowser; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styledComponents */ "./components/styledComponents.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/browsing/datasetsBrowsing/partBrowser.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;




var Option = antd__WEBPACK_IMPORTED_MODULE_1__["Select"].Option;
var PartsBrowser = function PartsBrowser(_ref) {
  _s();

  var setName = _ref.setName,
      setGroupBy = _ref.setGroupBy,
      resultsNames = _ref.resultsNames,
      restParts = _ref.restParts,
      part = _ref.part,
      name = _ref.name,
      setSelectedParts = _ref.setSelectedParts,
      selectedParts = _ref.selectedParts,
      selectedName = _ref.selectedName;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(name),
      value = _useState[0],
      setValue = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(false),
      openSelect = _useState2[0],
      setSelect = _useState2[1];

  return __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledSelect"], {
    dropdownMatchSelectWidth: false,
    defaultValue: name,
    selected: selectedName === value ? 'selected' : '',
    onChange: function onChange(value) {
      selectedParts[part] = value;
      setSelectedParts(selectedParts);
      setGroupBy(part);
      setValue(value);
      setName(value);
    },
    onClick: function onClick() {
      return setSelect(!openSelect);
    },
    open: openSelect,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 35,
      columnNumber: 5
    }
  }, resultsNames.map(function (result) {
    return __jsx(Option, {
      value: result,
      key: result,
      onClick: function onClick() {
        return setSelect(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 50,
        columnNumber: 9
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledOptionContent"], {
      availability: "available",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 51,
        columnNumber: 11
      }
    }, result));
  }), restParts.map(function (result) {
    return __jsx(Option, {
      key: result,
      value: result,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 57,
        columnNumber: 9
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledOptionContent"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 58,
        columnNumber: 11
      }
    }, result));
  }));
};

_s(PartsBrowser, "SbubPWqYQcO5mFa49acVznAtXOI=");

_c = PartsBrowser;

var _c;

$RefreshReg$(_c, "PartsBrowser");

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

/***/ }),

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

/***/ }),

/***/ "./components/browsing/lumesectionBroweser.tsx":
/*!*****************************************************!*\
  !*** ./components/browsing/lumesectionBroweser.tsx ***!
  \*****************************************************/
/*! exports provided: LumesectionBrowser */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LumesectionBrowser", function() { return LumesectionBrowser; });
/* harmony import */ var _babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/toConsumableArray */ "./node_modules/@babel/runtime/helpers/esm/toConsumableArray.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/browsing/lumesectionBroweser.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];







var Option = antd__WEBPACK_IMPORTED_MODULE_3__["Select"].Option;
var LumesectionBrowser = function LumesectionBrowser(_ref) {
  _s();

  var color = _ref.color,
      currentLumisection = _ref.currentLumisection,
      handler = _ref.handler,
      currentRunNumber = _ref.currentRunNumber,
      currentDataset = _ref.currentDataset;

  //0 - it represents ALL lumisections. If none lumisection is selected, then plots which are displaid
  //consist of ALL lumisections.
  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([{
    label: 'All',
    value: 0
  }]),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      lumisections = _React$useState2[0],
      setLumisections = _React$useState2[1];

  var current_time = new Date().getTime();

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](current_time),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      not_older_than = _React$useState4[0],
      set_not_older_than = _React$useState4[1]; //getting all run lumisections


  var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_6__["getLumisections"])({
    run_number: currentRunNumber,
    dataset_name: currentDataset,
    lumi: -1,
    notOlderThan: not_older_than
  }), {}, [currentRunNumber, currentDataset, not_older_than]),
      data = _useRequest.data,
      isLoading = _useRequest.isLoading,
      errors = _useRequest.errors;

  var all_runs_with_lumi = data ? data.data : [];
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    //extracting just lumisections from data object
    var lumisections_from_api = all_runs_with_lumi.length > 0 ? all_runs_with_lumi.map(function (run) {
      return {
        label: run.lumi.toString(),
        value: run.lumi
      };
    }) : [];

    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(lumisections);

    var allLumis = copy.concat(lumisections_from_api);
    setLumisections(allLumis);
  }, [all_runs_with_lumi]);
  var lumiValues = lumisections.map(function (lumi) {
    return lumi.value;
  }); //0 lumisection is not exists, it added as representation of ALL lumisections. If none of lumesctions is selected
  //it means that should be displaid plots which constist of ALL lumiections.
  //The same situation when run doesn't have lumis at all. It means that it displays plots of ALL Lumis

  var currentLumiIndex = lumiValues.indexOf(currentLumisection) === -1 ? 0 : lumiValues.indexOf(currentLumisection);
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_8__["StyledFormItem"], {
    labelcolor: color,
    name: 'lumi',
    label: "Lumi",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Row"], {
    justify: "center",
    align: "middle",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Button"], {
    disabled: !lumisections[currentLumiIndex - 1],
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["CaretLeftFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 87,
        columnNumber: 21
      }
    }),
    type: "link",
    onClick: function onClick() {
      handler(lumiValues[currentLumiIndex - 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 85,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 11
    }
  }, __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledSelect"], {
    dropdownMatchSelectWidth: false,
    value: lumiValues[currentLumiIndex],
    onChange: function onChange(e) {
      handler(parseInt(e));
    },
    showSearch: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
      columnNumber: 13
    }
  }, lumisections && lumisections.map(function (current_lumisection) {
    return __jsx(Option, {
      value: current_lumisection.value,
      key: current_lumisection.value.toString(),
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 107,
        columnNumber: 21
      }
    }, isLoading ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["OptionParagraph"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 112,
        columnNumber: 25
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Spin"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 113,
        columnNumber: 27
      }
    })) : __jsx("p", {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 116,
        columnNumber: 25
      }
    }, current_lumisection.label));
  }))), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 123,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Button"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["CaretRightFilled"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 125,
        columnNumber: 21
      }
    }),
    disabled: !lumisections[currentLumiIndex + 1],
    type: "link",
    onClick: function onClick() {
      handler(lumiValues[currentLumiIndex + 1]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 124,
      columnNumber: 13
    }
  })))));
};

_s(LumesectionBrowser, "q895zhX773y6tyrL44zL+52rjM0=", false, function () {
  return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__["useRequest"]];
});

_c = LumesectionBrowser;

var _c;

$RefreshReg$(_c, "LumesectionBrowser");

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

/***/ }),

/***/ "./components/menu.tsx":
/*!*****************************!*\
  !*** ./components/menu.tsx ***!
  \*****************************/
/*! exports provided: DropdownMenu */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DropdownMenu", function() { return DropdownMenu; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/menu.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;



var DropdownMenu = function DropdownMenu(_ref) {
  _s();

  var options = _ref.options,
      defaultValue = _ref.defaultValue,
      action = _ref.action;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_0__["useState"])(defaultValue),
      value = _useState[0],
      setValue = _useState[1];

  var plotMenu = function plotMenu(options, defaultValue) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 16,
        columnNumber: 5
      }
    }, options.map(function (option) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
        key: option.value,
        onClick: function onClick() {
          action && action(option.value);
          setValue(option);
        },
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 18,
          columnNumber: 9
        }
      }, __jsx("div", {
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 25,
          columnNumber: 11
        }
      }, option.label));
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options, defaultValue),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 34,
      columnNumber: 9
    }
  }, __jsx("a", {
    style: {
      color: 'white'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 35,
      columnNumber: 11
    }
  }, value.label, " ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 27
    }
  }), ' '))));
};

_s(DropdownMenu, "+di++irDcPspjmhJVP9frUKGzpo=");

_c = DropdownMenu;

var _c;

$RefreshReg$(_c, "DropdownMenu");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./components/navigation/archive_mode_header.tsx":
/*!*******************************************************!*\
  !*** ./components/navigation/archive_mode_header.tsx ***!
  \*******************************************************/
/*! exports provided: ArchiveModeHeader */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ArchiveModeHeader", function() { return ArchiveModeHeader; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _freeSearchResultModal__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./freeSearchResultModal */ "./components/navigation/freeSearchResultModal.tsx");
/* harmony import */ var _browsing__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../browsing */ "./components/browsing/index.tsx");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../searchButton */ "./components/searchButton.tsx");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/archive_mode_header.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];






var ArchiveModeHeader = function ArchiveModeHeader() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var run = query.run_number ? query.run_number : '';

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](run),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      search_run_number = _React$useState2[0],
      setSearchRunNumber = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_1__["useState"](query.dataset_name),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState3, 2),
      search_dataset_name = _React$useState4[0],
      setSearchDatasetName = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_1__["useState"](false),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState5, 2),
      modalState = _React$useState6[0],
      setModalState = _React$useState6[1];

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    //when modal is open, run number and dataset search fields are filled with values from query
    if (modalState) {
      var _run = query.run_number ? query.run_number : '';

      setSearchDatasetName(query.dataset_name);
      setSearchRunNumber(_run);
    }
  }, [modalState]);
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomCol"], {
    display: "flex",
    alignitems: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 5
    }
  }, __jsx(_freeSearchResultModal__WEBPACK_IMPORTED_MODULE_4__["SearchModal"], {
    modalState: modalState,
    setModalState: setModalState,
    setSearchRunNumber: setSearchRunNumber,
    setSearchDatasetName: setSearchDatasetName,
    search_run_number: search_run_number,
    search_dataset_name: search_dataset_name,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 7
    }
  }), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomRow"], {
    width: "fit-content",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 7
    }
  }, __jsx(_browsing__WEBPACK_IMPORTED_MODULE_5__["Browser"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_6__["SearchButton"], {
    onClick: function onClick() {
      return setModalState(true);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 9
    }
  })));
};

_s(ArchiveModeHeader, "4+byPcHoYQ6WxQuWVTcn1ISAZZI=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = ArchiveModeHeader;

var _c;

$RefreshReg$(_c, "ArchiveModeHeader");

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

/***/ }),

/***/ "./components/navigation/composedSearch.tsx":
/*!**************************************************!*\
  !*** ./components/navigation/composedSearch.tsx ***!
  \**************************************************/
/*! exports provided: ComposedSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ComposedSearch", function() { return ComposedSearch; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _workspaces__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../workspaces */ "./components/workspaces/index.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../plots/plot/plotSearch */ "./components/plots/plot/plotSearch/index.tsx");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _liveModeHeader__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./liveModeHeader */ "./components/navigation/liveModeHeader.tsx");
/* harmony import */ var _archive_mode_header__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./archive_mode_header */ "./components/navigation/archive_mode_header.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/composedSearch.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];









var ComposedSearch = function ComposedSearch() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var set_on_live_mode = query.run_number === '0' && query.dataset_name === '/Global/Online/ALL';
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomRow"], {
    width: "100%",
    display: "flex",
    justifycontent: "space-between",
    alignitems: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 21,
      columnNumber: 5
    }
  }, set_on_live_mode ? __jsx(_liveModeHeader__WEBPACK_IMPORTED_MODULE_7__["LiveModeHeader"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 28,
      columnNumber: 9
    }
  }) : __jsx(_archive_mode_header__WEBPACK_IMPORTED_MODULE_8__["ArchiveModeHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 9
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 9
    }
  }, __jsx(_workspaces__WEBPACK_IMPORTED_MODULE_3__["default"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 34,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 9
    }
  }, __jsx(_plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_5__["PlotSearch"], {
    isLoadingFolders: false,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 11
    }
  }))));
};

_s(ComposedSearch, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = ComposedSearch;

var _c;

$RefreshReg$(_c, "ComposedSearch");

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

/***/ }),

/***/ "./components/navigation/freeSearchResultModal.tsx":
/*!*********************************************************!*\
  !*** ./components/navigation/freeSearchResultModal.tsx ***!
  \*********************************************************/
/*! exports provided: SearchModal */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SearchModal", function() { return SearchModal; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! qs */ "./node_modules/qs/lib/index.js");
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(qs__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _containers_search_SearchResults__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../containers/search/SearchResults */ "./containers/search/SearchResults.tsx");
/* harmony import */ var _hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../hooks/useSearch */ "./hooks/useSearch.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _selectedData__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ./selectedData */ "./components/navigation/selectedData.tsx");
/* harmony import */ var _Nav__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../Nav */ "./components/Nav.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/freeSearchResultModal.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3___default.a.createElement;













var open_a_new_tab = function open_a_new_tab(query) {
  window.open(query, '_blank');
};

var SearchModal = function SearchModal(_ref) {
  _s();

  var setModalState = _ref.setModalState,
      modalState = _ref.modalState,
      search_run_number = _ref.search_run_number,
      search_dataset_name = _ref.search_dataset_name,
      setSearchDatasetName = _ref.setSearchDatasetName,
      setSearchRunNumber = _ref.setSearchRunNumber;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"])();
  var query = router.query;
  var dataset = query.dataset_name ? query.dataset_name : '';

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(dataset),
      datasetName = _useState[0],
      setDatasetName = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(false),
      openRunInNewTab = _useState2[0],
      toggleRunInNewTab = _useState2[1];

  var run = query.run_number ? query.run_number : '';

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(run),
      runNumber = _useState3[0],
      setRunNumber = _useState3[1];

  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    var run = query.run_number ? query.run_number : '';
    var dataset = query.dataset_name ? query.dataset_name : '';
    setDatasetName(dataset);
    setRunNumber(run);
  }, [query.dataset_name, query.run_number]);

  var onClosing = function onClosing() {
    setModalState(false);
  };

  var searchHandler = function searchHandler(run_number, dataset_name) {
    setDatasetName(dataset_name);
    setRunNumber(run_number);
  };

  var navigationHandler = function navigationHandler(search_by_run_number, search_by_dataset_name) {
    setSearchRunNumber(search_by_run_number);
    setSearchDatasetName(search_by_dataset_name);
  };

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__["useSearch"])(search_run_number, search_dataset_name),
      results_grouped = _useSearch.results_grouped,
      searching = _useSearch.searching,
      isLoading = _useSearch.isLoading,
      errors = _useSearch.errors;

  var onOk = /*#__PURE__*/function () {
    var _ref2 = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.mark(function _callee() {
      var params, new_tab_query_params, current_root;
      return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.wrap(function _callee$(_context) {
        while (1) {
          switch (_context.prev = _context.next) {
            case 0:
              if (!openRunInNewTab) {
                _context.next = 7;
                break;
              }

              params = form.getFieldsValue();
              new_tab_query_params = qs__WEBPACK_IMPORTED_MODULE_4___default.a.stringify(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_14__["getChangedQueryParams"])(params, query)); //root url is ends with first '?'. I can't use just root url from config.config, because
              //in dev env it use localhost:8081/dqm/dev (this is old backend url from where I'm getting data),
              //but I need localhost:3000

              current_root = window.location.href.split('/?')[0];
              open_a_new_tab("".concat(current_root, "/?").concat(new_tab_query_params));
              _context.next = 9;
              break;

            case 7:
              _context.next = 9;
              return form.submit();

            case 9:
              onClosing();

            case 10:
            case "end":
              return _context.stop();
          }
        }
      }, _callee);
    }));

    return function onOk() {
      return _ref2.apply(this, arguments);
    };
  }();

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_6__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  return __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Search data",
    visible: modalState,
    onCancel: function onCancel() {
      return onClosing();
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_11__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return onClosing();
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 104,
        columnNumber: 9
      }
    }, "Close"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      key: "OK",
      onClick: onOk,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 112,
        columnNumber: 9
      }
    }, "OK")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 5
    }
  }, modalState && __jsx(react__WEBPACK_IMPORTED_MODULE_3___default.a.Fragment, null, __jsx(_Nav__WEBPACK_IMPORTED_MODULE_13__["default"], {
    initial_search_run_number: search_run_number,
    initial_search_dataset_name: search_dataset_name,
    defaultDatasetName: datasetName,
    defaultRunNumber: runNumber,
    handler: navigationHandler,
    type: "top",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 11
    }
  }), __jsx(_selectedData__WEBPACK_IMPORTED_MODULE_12__["SelectedData"], {
    form: form,
    dataset_name: datasetName,
    run_number: runNumber,
    toggleRunInNewTab: toggleRunInNewTab,
    openRunInNewTab: openRunInNewTab,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 127,
      columnNumber: 11
    }
  }), searching ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 135,
      columnNumber: 13
    }
  }, __jsx(_containers_search_SearchResults__WEBPACK_IMPORTED_MODULE_8__["default"], {
    handler: searchHandler,
    isLoading: isLoading,
    results_grouped: results_grouped,
    errors: errors,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 136,
      columnNumber: 15
    }
  })) : __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 144,
      columnNumber: 13
    }
  })));
};

_s(SearchModal, "cJSZLTqxYxam8F0Rr2yyVtEoUY8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"], _hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__["useSearch"], antd__WEBPACK_IMPORTED_MODULE_6__["Form"].useForm];
});

_c = SearchModal;

var _c;

$RefreshReg$(_c, "SearchModal");

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

/***/ }),

/***/ "./components/navigation/liveModeHeader.tsx":
/*!**************************************************!*\
  !*** ./components/navigation/liveModeHeader.tsx ***!
  \**************************************************/
/*! exports provided: LiveModeHeader */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LiveModeHeader", function() { return LiveModeHeader; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/liveModeHeader.tsx",
    _s2 = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];












var Title = antd__WEBPACK_IMPORTED_MODULE_1__["Typography"].Title;
var LiveModeHeader = function LiveModeHeader(_ref) {
  _s2();

  var _s = $RefreshSig$();

  var query = _ref.query;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__["useUpdateLiveMode"])(),
      update = _useUpdateLiveMode.update,
      set_update = _useUpdateLiveMode.set_update;

  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]);
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomForm"], {
    display: "flex",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 7
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_8__["main_run_info"].map(_s(function (info) {
    _s();

    var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["FormatParamsForAPI"])(globalState, query, info.value, '/HLT/EventInfo');

    var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_10__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number]),
        data = _useRequest.data,
        isLoading = _useRequest.isLoading;

    return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CutomFormItem"], {
      space: "8",
      width: "fit-content",
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.common.white,
      name: info.label,
      label: info.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 46,
        columnNumber: 13
      }
    }, __jsx(Title, {
      level: 4,
      style: {
        color: "".concat(update ? _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.error)
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 53,
        columnNumber: 15
      }
    }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
      size: "small",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 63,
        columnNumber: 30
      }
    }) : Object(_utils__WEBPACK_IMPORTED_MODULE_11__["get_label"])(info, data)));
  }, "4RN8DXN8bS1gZHtH2GHRXx1u2KI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_9__["useRequest"]];
  }))), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomCol"], {
    justifycontent: "flex-end",
    display: "flex",
    alignitems: "center",
    texttransform: "uppercase",
    color: update ? _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.success : _styles_theme__WEBPACK_IMPORTED_MODULE_4__["theme"].colors.notification.error,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 7
    }
  }, "Live Mode", __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
    space: "2",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Tooltip"], {
    title: "Updating mode is ".concat(update ? 'on' : 'off'),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "primary",
    shape: "circle",
    onClick: function onClick() {
      set_update(!update);
    },
    icon: update ? __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["PauseOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 89,
        columnNumber: 30
      }
    }) : __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["PlayCircleOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 89,
        columnNumber: 50
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 13
    }
  })))));
};

_s2(LiveModeHeader, "/GgdwDg0hgmNajas7INOw5S4hBA=", false, function () {
  return [_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_5__["useUpdateLiveMode"]];
});

_c = LiveModeHeader;

var _c;

$RefreshReg$(_c, "LiveModeHeader");

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

/***/ }),

/***/ "./components/navigation/selectedData.tsx":
/*!************************************************!*\
  !*** ./components/navigation/selectedData.tsx ***!
  \************************************************/
/*! exports provided: SelectedData */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SelectedData", function() { return SelectedData; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _browsing_lumesectionBroweser__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../browsing/lumesectionBroweser */ "./components/browsing/lumesectionBroweser.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _radioButtonsGroup__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../radioButtonsGroup */ "./components/radioButtonsGroup.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/selectedData.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];










var open_in_a_new_tab_options = [{
  value: true,
  label: 'Yes'
}, {
  value: false,
  label: 'No'
}];
var SelectedData = function SelectedData(_ref) {
  _s();

  var dataset_name = _ref.dataset_name,
      run_number = _ref.run_number,
      form = _ref.form,
      toggleRunInNewTab = _ref.toggleRunInNewTab,
      openRunInNewTab = _ref.openRunInNewTab;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__["store"]),
      lumisection = _React$useContext.lumisection,
      setLumisection = _React$useContext.setLumisection;

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_8__["useRouter"])();
  var query = router.query;

  var lumisectionsChangeHandler = function lumisectionsChangeHandler(lumi) {
    //we set lumisection in inseContext store in order to save a it's value.
    //When form is submitted(onFinish...)(clicked button "OK" in dialog), then
    //url is changed
    setLumisection(lumi);
  };

  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_4___default.a, {
    form: form,
    onFinish: function onFinish(params) {
      //when OK is clicked, run number, dataset and lumi params in url is changed.
      Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_7__["changeRouter"])(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_7__["getChangedQueryParams"])(params, query));
    },
    fields: [{
      name: 'dataset_name',
      value: dataset_name
    }, {
      name: 'run_number',
      value: run_number
    }, {
      name: 'lumi',
      value: lumisection
    }],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 5
    }
  }, __jsx("hr", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 7
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledFormItem"], {
    name: 'dataset_name',
    label: "Dataset name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 71,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["SelectedDataCol"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 11
    }
  }, dataset_name))), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 7
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledFormItem"], {
    name: 'run_number',
    label: "Run number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["SelectedDataCol"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 77,
      columnNumber: 11
    }
  }, run_number))), _config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].new_back_end.lumisections_on && __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 11
    }
  }, __jsx(_browsing_lumesectionBroweser__WEBPACK_IMPORTED_MODULE_3__["LumesectionBrowser"], {
    color: "black",
    handler: lumisectionsChangeHandler,
    currentLumisection: lumisection,
    currentDataset: dataset_name,
    currentRunNumber: run_number,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 13
    }
  }))), __jsx("hr", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 94,
      columnNumber: 7
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledFormItem"], {
    name: 'open_in_a_new_a_tab',
    label: "Open in a new tab?",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 9
    }
  }, __jsx(_radioButtonsGroup__WEBPACK_IMPORTED_MODULE_9__["RadioButtonsGroup"], {
    options: open_in_a_new_tab_options,
    getOptionLabel: function getOptionLabel(option) {
      return option.label;
    },
    getOptionValue: function getOptionValue(option) {
      return option.value;
    },
    current_value: openRunInNewTab,
    action: function action(value) {
      toggleRunInNewTab(value);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 96,
      columnNumber: 11
    }
  }))), __jsx("hr", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 107,
      columnNumber: 7
    }
  }));
};

_s(SelectedData, "aMl+k4nuJg9qVuzIcxJt+MRrKSw=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_8__["useRouter"]];
});

_c = SelectedData;

var _c;

$RefreshReg$(_c, "SelectedData");

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

/***/ }),

/***/ "./components/plots/plot/plotSearch/index.tsx":
/*!****************************************************!*\
  !*** ./components/plots/plot/plotSearch/index.tsx ***!
  \****************************************************/
/*! exports provided: PlotSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotSearch", function() { return PlotSearch; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../containers/display/utils */ "./containers/display/utils.ts");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/plot/plotSearch/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];





var PlotSearch = function PlotSearch(_ref) {
  _s();

  var isLoadingFolders = _ref.isLoadingFolders;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](query.plot_search),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      plotName = _React$useState2[0],
      setPlotName = _React$useState2[1];

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (query.plot_search !== plotName) {
      var params = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["getChangedQueryParams"])({
        plot_search: plotName
      }, query);
      Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["changeRouter"])(params);
    }
  }, [plotName]);
  return react__WEBPACK_IMPORTED_MODULE_1__["useMemo"](function () {
    return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default.a, {
      onChange: function onChange(e) {
        return setPlotName(e.target.value);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 32,
        columnNumber: 7
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 33,
        columnNumber: 9
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSearch"], {
      defaultValue: query.plot_search,
      loading: isLoadingFolders,
      id: "plot_search",
      placeholder: "Enter plot name",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 34,
        columnNumber: 11
      }
    })));
  }, [plotName]);
};

_s(PlotSearch, "qUuwOtWUsWURNKw3w2PjYEO5WgU=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"]];
});

_c = PlotSearch;

var _c;

$RefreshReg$(_c, "PlotSearch");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./components/runInfo/index.tsx":
/*!**************************************!*\
  !*** ./components/runInfo/index.tsx ***!
  \**************************************/
/*! exports provided: RunInfo */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "RunInfo", function() { return RunInfo; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _info__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../info */ "./components/info.tsx");
/* harmony import */ var _runInfoModal__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./runInfoModal */ "./components/runInfo/runInfoModal.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/runInfo/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];




var RunInfo = function RunInfo(_ref) {
  _s();

  var query = _ref.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      open = _React$useState2[0],
      toggleModal = _React$useState2[1];

  return __jsx(react__WEBPACK_IMPORTED_MODULE_1__["Fragment"], null, __jsx(_runInfoModal__WEBPACK_IMPORTED_MODULE_3__["RunInfoModal"], {
    toggleModal: toggleModal,
    open: open,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 17,
      columnNumber: 7
    }
  }), __jsx("div", {
    onClick: function onClick() {
      return toggleModal(!open);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 18,
      columnNumber: 7
    }
  }, __jsx(_info__WEBPACK_IMPORTED_MODULE_2__["Info"], {
    content: 'Run info',
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 19,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["RunInfoIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 20,
      columnNumber: 11
    }
  }))));
};

_s(RunInfo, "eaX+I6gnmnbIQARWL5M4Rbyi1Ow=");

_c = RunInfo;

var _c;

$RefreshReg$(_c, "RunInfo");

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

/***/ }),

/***/ "./components/runInfo/runInfoModal.tsx":
/*!*********************************************!*\
  !*** ./components/runInfo/runInfoModal.tsx ***!
  \*********************************************/
/*! exports provided: RunInfoModal */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "RunInfoModal", function() { return RunInfoModal; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _runStartTimeStamp__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./runStartTimeStamp */ "./components/runInfo/runStartTimeStamp.tsx");
/* harmony import */ var _constants__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../constants */ "./components/constants.ts");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../utils */ "./components/utils.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/runInfo/runInfoModal.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];











var RunInfoModal = function RunInfoModal(_ref) {
  _s();

  var query = _ref.query,
      toggleModal = _ref.toggleModal,
      open = _ref.open;
  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]);
  var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_6__["FormatParamsForAPI"])(globalState, query, 'iRun', '/HLT/EventInfo');

  var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_9__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number]),
      data = _useRequest.data,
      isLoading = _useRequest.isLoading;

  var run = Object(_utils__WEBPACK_IMPORTED_MODULE_10__["get_label"])({
    value: 'iRun',
    label: 'Run'
  }, data);
  return __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_1__["StyledModal"], {
    title: "Run ".concat(run, " information"),
    visible: open,
    onCancel: function onCancel() {
      return toggleModal(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_3__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleModal(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 46,
        columnNumber: 9
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 5
    }
  }, open && __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 9
    }
  }, _constants__WEBPACK_IMPORTED_MODULE_5__["run_info"].map(function (info) {
    return __jsx(_runStartTimeStamp__WEBPACK_IMPORTED_MODULE_4__["RunInfoItem"], {
      info: info,
      query: query,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 59,
        columnNumber: 13
      }
    });
  })));
};

_s(RunInfoModal, "Ltjcf9IDO6m8Vn5x75FIfH0pTBs=", false, function () {
  return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
});

_c = RunInfoModal;

var _c;

$RefreshReg$(_c, "RunInfoModal");

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

/***/ }),

/***/ "./components/runInfo/runStartTimeStamp.tsx":
/*!**************************************************!*\
  !*** ./components/runInfo/runStartTimeStamp.tsx ***!
  \**************************************************/
/*! exports provided: RunInfoItem */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "RunInfoItem", function() { return RunInfoItem; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../plots/plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/runInfo/runStartTimeStamp.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];







var RunInfoItem = function RunInfoItem(_ref) {
  _s();

  var query = _ref.query,
      info = _ref.info;
  var globalState = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_4__["store"]);
  var params_for_api = Object(_plots_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_5__["FormatParamsForAPI"])(globalState, query, info.value, '/HLT/EventInfo');

  var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_2__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_3__["get_jroot_plot"])(params_for_api), {}, [query.dataset_name, query.run_number]),
      data = _useRequest.data,
      isLoading = _useRequest.isLoading;

  var get_label = function get_label(info) {
    var value = data ? data.fString : null;

    if (info.type === 'time' && value) {
      var milisec = new Date(parseInt(value) * 1000);
      var time = milisec.toUTCString();
      return time;
    } else {
      return value ? value : 'No information';
    }
  };

  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_6__["CustomRow"], {
    display: "flex",
    justifycontent: "space-between",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_6__["CustomCol"], {
    space: '1',
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 7
    }
  }, info.label), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_6__["CustomCol"], {
    space: '1',
    bold: "true",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 7
    }
  }, isLoading ? __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Spin"], {
    size: "small",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 22
    }
  }) : get_label(info)));
};

_s(RunInfoItem, "Ltjcf9IDO6m8Vn5x75FIfH0pTBs=", false, function () {
  return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_2__["useRequest"]];
});

_c = RunInfoItem;

var _c;

$RefreshReg$(_c, "RunInfoItem");

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

/***/ }),

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/workspaces/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];












var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__["store"]),
      workspace = _React$useContext.workspace,
      setWorkspace = _React$useContext.setWorkspace;

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var initialWorkspace = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? workspaces[0].workspaces[1].label : workspaces[0].workspaces[3].label;
  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(initialWorkspace);
    return function () {
      return setWorkspace(initialWorkspace);
    };
  }, []);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1]; // make a workspace set from context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_12__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleWorkspaces(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 57,
        columnNumber: 13
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 67,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 69,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 71,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "9wsb3E7mFlyFmQpi1Uvfk2BcVak=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

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

/***/ }),

/***/ "./containers/display/header.tsx":
/*!***************************************!*\
  !*** ./containers/display/header.tsx ***!
  \***************************************/
/*! exports provided: Header */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Header", function() { return Header; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _components_runInfo__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../components/runInfo */ "./components/runInfo/index.tsx");
/* harmony import */ var _components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../components/navigation/composedSearch */ "./components/navigation/composedSearch.tsx");
/* harmony import */ var _components_Nav__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../components/Nav */ "./components/Nav.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/display/header.tsx";

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var Header = function Header(_ref) {
  var isDatasetAndRunNumberSelected = _ref.isDatasetAndRunNumberSelected,
      query = _ref.query;
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, //if all full set is selected: dataset name and run number, then regular search field is not visible.
  //Instead, run and dataset browser is is displayed.
  //Regular search fields are displayed just in the main page.
  isDatasetAndRunNumberSelected ? __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_runInfo__WEBPACK_IMPORTED_MODULE_2__["RunInfo"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 26,
      columnNumber: 13
    }
  }), __jsx(_components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__["ComposedSearch"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 27,
      columnNumber: 13
    }
  })) : __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_Nav__WEBPACK_IMPORTED_MODULE_4__["default"], {
    initial_search_run_number: query.search_run_number,
    initial_search_dataset_name: query.search_dataset_name,
    initial_search_lumisection: query.lumi,
    handler: _utils_pages__WEBPACK_IMPORTED_MODULE_1__["navigationHandler"],
    type: "top",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 13
    }
  })));
};
_c = Header;

var _c;

$RefreshReg$(_c, "Header");

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

/***/ }),

/***/ "./hooks/useAvailbleAndNotAvailableDatasetPartsOptions.tsx":
/*!*****************************************************************!*\
  !*** ./hooks/useAvailbleAndNotAvailableDatasetPartsOptions.tsx ***!
  \*****************************************************************/
/*! exports provided: useAvailbleAndNotAvailableDatasetPartsOptions */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useAvailbleAndNotAvailableDatasetPartsOptions", function() { return useAvailbleAndNotAvailableDatasetPartsOptions; });
/* harmony import */ var _babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/defineProperty */ "./node_modules/@babel/runtime/helpers/esm/defineProperty.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _useSearch__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./useSearch */ "./hooks/useSearch.tsx");
/* harmony import */ var _components_viewDetailsMenu_utils__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../components/viewDetailsMenu/utils */ "./components/viewDetailsMenu/utils.ts");
/* harmony import */ var _components_browsing_utils__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../components/browsing/utils */ "./components/browsing/utils.ts");


var _s = $RefreshSig$();





var useAvailbleAndNotAvailableDatasetPartsOptions = function useAvailbleAndNotAvailableDatasetPartsOptions(run_number, currentDataset) {
  _s();

  var selectedDatasetParts = Object(_components_browsing_utils__WEBPACK_IMPORTED_MODULE_4__["getDatasetNameSplitBySlashIntoObject"])(currentDataset);
  var datasetPartsPositions = Object.keys(selectedDatasetParts).sort();

  var _useSearch = Object(_useSearch__WEBPACK_IMPORTED_MODULE_2__["useSearch"])(run_number, ''),
      results_grouped = _useSearch.results_grouped; //allDatasets are all possible datasets


  var allDatasets = results_grouped.map(function (result) {
    return result.dataset;
  });
  var firstPosition = datasetPartsPositions[0]; //lastSelectedDatasetPartPosition: is POSITION of last selected dataset part
  //lastSelectedDatasetPartPosition is use for grouping all dataset parts possible variants.

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(firstPosition),
      lastSelectedDatasetPartPosition = _useState[0],
      setLastSelectedDatasetPartPosition = _useState[1]; //lastSelectedDatasetPartOption: is VALUE of last selected dataset part


  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(selectedDatasetParts[firstPosition]),
      lastSelectedDatasetPartValue = _useState2[0],
      setLastSelectedDatasetPartValue = _useState2[1]; //selectedParts: is SLECTED dataset parts, from whom could be formed full datasetname
  // by defaut selectedParts is formed from currentDataset


  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(Object(_components_browsing_utils__WEBPACK_IMPORTED_MODULE_4__["getDatasetNameSplitBySlashIntoObject"])(currentDataset)),
      selectedParts = _useState3[0],
      setSelectedParts = _useState3[1]; //allDatasets is string array. One string from this array is FULL dataset name. We need to
  //separated each dataset name to parts. One part of dataset name in FULL string is separated by slash.
  //getDatasetParts function separates dataset names to parts and group them by LAST SELECTED DATASET PART POSITION.
  //getDatasetParts returns LAST SELECTED POSITION VALUE and it possible combinations with other parts


  var partsObjectArray = Object(_components_viewDetailsMenu_utils__WEBPACK_IMPORTED_MODULE_3__["getDatasetParts"])(allDatasets, lastSelectedDatasetPartPosition); //from all selected dataset name's parts we form full dataset name.
  //Values of selected dataset parts are in datasetParts array
  //The first element of array is empty string, because dataset name should start with slash.

  var datasetParts = Object.values(selectedParts);
  datasetParts.unshift('');
  var fullDatasetName = datasetParts.join('/'); //We check is dataset name combined from parts is exits in all possible dataset names.
  // rename doesCombinationOfSelectedDatasetPartsExists to datasetExists or resultingDatasetNameCombinationExists

  var doesCombinationOfSelectedDatasetPartsExists = allDatasets.includes(fullDatasetName);
  var availableAndNotAvailableDatasetParts = datasetPartsPositions.map(function (part) {
    var availableChoices = Object(_components_browsing_utils__WEBPACK_IMPORTED_MODULE_4__["getAvailableChoices"])(partsObjectArray, lastSelectedDatasetPartValue, part);
    var notAvailableChoices = Object(_components_browsing_utils__WEBPACK_IMPORTED_MODULE_4__["getRestOptions"])(availableChoices, allDatasets, part);
    return Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__["default"])({}, part, {
      availableChoices: availableChoices,
      notAvailableChoices: notAvailableChoices
    });
  });
  return {
    availableAndNotAvailableDatasetParts: availableAndNotAvailableDatasetParts,
    setSelectedParts: setSelectedParts,
    selectedParts: selectedParts,
    setLastSelectedDatasetPartValue: setLastSelectedDatasetPartValue,
    lastSelectedDatasetPartValue: lastSelectedDatasetPartValue,
    setLastSelectedDatasetPartPosition: setLastSelectedDatasetPartPosition,
    doesCombinationOfSelectedDatasetPartsExists: doesCombinationOfSelectedDatasetPartsExists,
    fullDatasetName: fullDatasetName
  };
};

_s(useAvailbleAndNotAvailableDatasetPartsOptions, "u98eYS81NgPgzqqTCtq+Ey5WPhk=", false, function () {
  return [_useSearch__WEBPACK_IMPORTED_MODULE_2__["useSearch"]];
});

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./pages/index.tsx":
/*!*************************!*\
  !*** ./pages/index.tsx ***!
  \*************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/head */ "./node_modules/next/dist/next-server/lib/head.js");
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styles/styledComponents */ "./styles/styledComponents.ts");
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _containers_display_header__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../containers/display/header */ "./containers/display/header.tsx");
/* harmony import */ var _containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../containers/display/content/constent_switching */ "./containers/display/content/constent_switching.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/pages/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;









var Index = function Index() {
  _s();

  // We grab the query from the URL:
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var isDatasetAndRunNumberSelected = !!query.run_number && !!query.dataset_name;
  return __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 29,
      columnNumber: 5
    }
  }, __jsx(next_head__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 7
    }
  }, __jsx("script", {
    crossOrigin: "anonymous",
    type: "text/javascript",
    src: "./jsroot-5.8.0/scripts/JSRootCore.js?2d&hist&more2d",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 9
    }
  })), __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLayout"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 7
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: "Back to main page",
    placement: "bottomLeft",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 11
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 13
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoWrapper"], {
    onClick: function onClick(e) {
      return Object(_utils_pages__WEBPACK_IMPORTED_MODULE_5__["backToMainPage"])(e);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 15
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogo"], {
    src: "./images/CMSlogo_white_red_nolabel_1024_May2014.png",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 17
    }
  })))), __jsx(_containers_display_header__WEBPACK_IMPORTED_MODULE_6__["Header"], {
    isDatasetAndRunNumberSelected: isDatasetAndRunNumberSelected,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 11
    }
  })), __jsx(_containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__["ContentSwitching"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 9
    }
  })));
};

_s(Index, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = Index;
/* harmony default export */ __webpack_exports__["default"] = (Index);

var _c;

$RefreshReg$(_c, "Index");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./workspaces/online.ts":
/*!******************************!*\
  !*** ./workspaces/online.ts ***!
  \******************************/
/*! exports provided: summariesWorkspace, workspaces */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "summariesWorkspace", function() { return summariesWorkspace; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "workspaces", function() { return workspaces; });
var summariesWorkspace = [{
  label: 'Summary',
  foldersPath: ['Summary']
}, // {
//   label: 'Reports',
//   foldersPath: []
// },
{
  label: 'Shift',
  foldersPath: ['00 Shift']
}, {
  label: 'Info',
  foldersPath: ['Info']
}, // {
//   label: 'Certification',
//   foldersPath: []
// },
{
  label: 'Everything',
  foldersPath: []
}];
var triggerWorkspace = [{
  label: 'L1T',
  foldersPath: ['L1T']
}, {
  label: 'L1T2016EMU',
  foldersPath: ['L1T2016EMU']
}, {
  label: 'L1T2016',
  foldersPath: ['L1T2016']
}, {
  label: 'L1TEMU',
  foldersPath: ['L1TEMU']
}, {
  label: 'HLT',
  foldersPath: ['HLT']
}];
var trackerWorkspace = [{
  label: 'PixelPhase1',
  foldersPath: ['PixelPhase1']
}, {
  label: 'Pixel',
  foldersPath: ['Pixel']
}, {
  label: 'SiStrip',
  foldersPath: ['SiStrip', 'Tracking']
}];
var calorimetersWorkspace = [{
  label: 'Ecal',
  foldersPath: ['Ecal', 'EcalBarrel', 'EcalEndcap', 'EcalCalibration']
}, {
  label: 'EcalPreshower',
  foldersPath: ['EcalPreshower']
}, {
  label: 'HCAL',
  foldersPath: ['Hcal', 'Hcal2']
}, {
  label: 'HCALcalib',
  foldersPath: ['HcalCalib']
}, {
  label: 'Castor',
  foldersPath: ['Castor']
}];
var mounsWorkspace = [{
  label: 'CSC',
  foldersPath: ['CSC']
}, {
  label: 'DT',
  foldersPath: ['DT']
}, {
  label: 'RPC',
  foldersPath: ['RPC']
}];
var cttpsWorspace = [{
  label: 'TrackingStrip',
  foldersPath: ['CTPPS/TrackingStrip', 'CTPPS/common', 'CTPPS/TrackingStrip/Layouts']
}, {
  label: 'TrackingPixel',
  foldersPath: ['CTPPS/TrackingPixel', 'CTPPS/common', 'CTPPS/TrackingPixel/Layouts']
}, {
  label: 'TimingDiamond',
  foldersPath: ['CTPPS/TimingDiamond', 'CTPPS/common', 'CTPPS/TimingDiamond/Layouts']
}, {
  label: 'TimingFastSilicon',
  foldersPath: ['CTPPS/TimingFastSilicon', 'CTPPS/common', 'CTPPS/TimingFastSilicon/Layouts']
}];
var workspaces = [{
  label: 'Summaries',
  workspaces: summariesWorkspace
}, {
  label: 'Trigger',
  workspaces: triggerWorkspace
}, {
  label: 'Tracker',
  workspaces: trackerWorkspace
}, {
  label: 'Calorimeters',
  workspaces: calorimetersWorkspace
}, {
  label: 'Muons',
  workspaces: mounsWorkspace
}, {
  label: 'CTPPS',
  workspaces: cttpsWorspace
}];

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9icm93c2luZy9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXROYW1lQnVpbGRlci50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL2NvbXBvbmVudHMvYnJvd3NpbmcvZGF0YXNldHNCcm93c2luZy9wYXJ0QnJvd3Nlci50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL2NvbXBvbmVudHMvYnJvd3NpbmcvaW5kZXgudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL2Jyb3dzaW5nL2x1bWVzZWN0aW9uQnJvd2VzZXIudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL21lbnUudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL25hdmlnYXRpb24vYXJjaGl2ZV9tb2RlX2hlYWRlci50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL2NvbXBvbmVudHMvbmF2aWdhdGlvbi9jb21wb3NlZFNlYXJjaC50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL2NvbXBvbmVudHMvbmF2aWdhdGlvbi9mcmVlU2VhcmNoUmVzdWx0TW9kYWwudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL25hdmlnYXRpb24vbGl2ZU1vZGVIZWFkZXIudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL25hdmlnYXRpb24vc2VsZWN0ZWREYXRhLnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RTZWFyY2gvaW5kZXgudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL3J1bkluZm8vaW5kZXgudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL3J1bkluZm8vcnVuSW5mb01vZGFsLnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9ydW5JbmZvL3J1blN0YXJ0VGltZVN0YW1wLnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2hlYWRlci50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL2hvb2tzL3VzZUF2YWlsYmxlQW5kTm90QXZhaWxhYmxlRGF0YXNldFBhcnRzT3B0aW9ucy50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL3BhZ2VzL2luZGV4LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vd29ya3NwYWNlcy9vbmxpbmUudHMiXSwibmFtZXMiOlsiRGF0YXNldHNCdWlsZGVyIiwiY3VycmVudERhdGFzZXQiLCJxdWVyeSIsImN1cnJlbnRSdW5OdW1iZXIiLCJ1c2VBdmFpbGJsZUFuZE5vdEF2YWlsYWJsZURhdGFzZXRQYXJ0c09wdGlvbnMiLCJhdmFpbGFibGVBbmROb3RBdmFpbGFibGVEYXRhc2V0UGFydHMiLCJzZXRTZWxlY3RlZFBhcnRzIiwic2VsZWN0ZWRQYXJ0cyIsInNldExhc3RTZWxlY3RlZERhdGFzZXRQYXJ0VmFsdWUiLCJsYXN0U2VsZWN0ZWREYXRhc2V0UGFydFZhbHVlIiwic2V0TGFzdFNlbGVjdGVkRGF0YXNldFBhcnRQb3NpdGlvbiIsImRvZXNDb21iaW5hdGlvbk9mU2VsZWN0ZWREYXRhc2V0UGFydHNFeGlzdHMiLCJmdWxsRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJjaGFuZ2VSb3V0ZXIiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJkYXRhc2V0X25hbWUiLCJtYXAiLCJwYXJ0IiwicGFydE5hbWUiLCJPYmplY3QiLCJrZXlzIiwibm90QXZhaWxhYmxlQ2hvaWNlcyIsImF2YWlsYWJsZUNob2ljZXMiLCJPcHRpb24iLCJTZWxlY3QiLCJQYXJ0c0Jyb3dzZXIiLCJzZXROYW1lIiwic2V0R3JvdXBCeSIsInJlc3VsdHNOYW1lcyIsInJlc3RQYXJ0cyIsIm5hbWUiLCJzZWxlY3RlZE5hbWUiLCJ1c2VTdGF0ZSIsInZhbHVlIiwic2V0VmFsdWUiLCJvcGVuU2VsZWN0Iiwic2V0U2VsZWN0IiwicmVzdWx0IiwiQnJvd3NlciIsImRhdGFTZXRTZWxlY3Rpb25zIiwiZGF0YXNldE9wdGlvbiIsInNldERhdGFzZXRPcHRpb24iLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJydW5fbnVtYmVyIiwibHVtaSIsInBhcnNlSW50IiwiTmFOIiwiUmVhY3QiLCJ1c2VDb250ZXh0Iiwic3RvcmUiLCJzZXRMdW1pc2VjdGlvbiIsInNldEN1cnJlbnRSdW5OdW1iZXIiLCJzZXRDdXJyZW50RGF0YXNldCIsImx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIiwiTHVtZXNlY3Rpb25Ccm93c2VyIiwiY29sb3IiLCJjdXJyZW50THVtaXNlY3Rpb24iLCJoYW5kbGVyIiwibGFiZWwiLCJsdW1pc2VjdGlvbnMiLCJzZXRMdW1pc2VjdGlvbnMiLCJjdXJyZW50X3RpbWUiLCJEYXRlIiwiZ2V0VGltZSIsIm5vdF9vbGRlcl90aGFuIiwic2V0X25vdF9vbGRlcl90aGFuIiwidXNlUmVxdWVzdCIsImdldEx1bWlzZWN0aW9ucyIsIm5vdE9sZGVyVGhhbiIsImRhdGEiLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJhbGxfcnVuc193aXRoX2x1bWkiLCJsdW1pc2VjdGlvbnNfZnJvbV9hcGkiLCJsZW5ndGgiLCJydW4iLCJ0b1N0cmluZyIsImNvcHkiLCJhbGxMdW1pcyIsImNvbmNhdCIsImx1bWlWYWx1ZXMiLCJjdXJyZW50THVtaUluZGV4IiwiaW5kZXhPZiIsImUiLCJjdXJyZW50X2x1bWlzZWN0aW9uIiwiRHJvcGRvd25NZW51Iiwib3B0aW9ucyIsImRlZmF1bHRWYWx1ZSIsImFjdGlvbiIsInBsb3RNZW51Iiwib3B0aW9uIiwiQXJjaGl2ZU1vZGVIZWFkZXIiLCJzZWFyY2hfcnVuX251bWJlciIsInNldFNlYXJjaFJ1bk51bWJlciIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRTZWFyY2hEYXRhc2V0TmFtZSIsIm1vZGFsU3RhdGUiLCJzZXRNb2RhbFN0YXRlIiwiQ29tcG9zZWRTZWFyY2giLCJzZXRfb25fbGl2ZV9tb2RlIiwib3Blbl9hX25ld190YWIiLCJ3aW5kb3ciLCJvcGVuIiwiU2VhcmNoTW9kYWwiLCJkYXRhc2V0IiwiZGF0YXNldE5hbWUiLCJzZXREYXRhc2V0TmFtZSIsIm9wZW5SdW5Jbk5ld1RhYiIsInRvZ2dsZVJ1bkluTmV3VGFiIiwicnVuTnVtYmVyIiwic2V0UnVuTnVtYmVyIiwib25DbG9zaW5nIiwic2VhcmNoSGFuZGxlciIsIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwib25PayIsInBhcmFtcyIsImZvcm0iLCJnZXRGaWVsZHNWYWx1ZSIsIm5ld190YWJfcXVlcnlfcGFyYW1zIiwicXMiLCJzdHJpbmdpZnkiLCJjdXJyZW50X3Jvb3QiLCJsb2NhdGlvbiIsImhyZWYiLCJzcGxpdCIsInN1Ym1pdCIsIkZvcm0iLCJ1c2VGb3JtIiwidGhlbWUiLCJjb2xvcnMiLCJzZWNvbmRhcnkiLCJtYWluIiwiVGl0bGUiLCJUeXBvZ3JhcGh5IiwiTGl2ZU1vZGVIZWFkZXIiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInVwZGF0ZSIsInNldF91cGRhdGUiLCJnbG9iYWxTdGF0ZSIsIm1haW5fcnVuX2luZm8iLCJpbmZvIiwicGFyYW1zX2Zvcl9hcGkiLCJGb3JtYXRQYXJhbXNGb3JBUEkiLCJnZXRfanJvb3RfcGxvdCIsImNvbW1vbiIsIndoaXRlIiwibm90aWZpY2F0aW9uIiwic3VjY2VzcyIsImVycm9yIiwiZ2V0X2xhYmVsIiwib3Blbl9pbl9hX25ld190YWJfb3B0aW9ucyIsIlNlbGVjdGVkRGF0YSIsImx1bWlzZWN0aW9uIiwiUGxvdFNlYXJjaCIsImlzTG9hZGluZ0ZvbGRlcnMiLCJwbG90X3NlYXJjaCIsInBsb3ROYW1lIiwic2V0UGxvdE5hbWUiLCJ0YXJnZXQiLCJSdW5JbmZvIiwidG9nZ2xlTW9kYWwiLCJSdW5JbmZvTW9kYWwiLCJydW5faW5mbyIsIlJ1bkluZm9JdGVtIiwiZlN0cmluZyIsInR5cGUiLCJtaWxpc2VjIiwidGltZSIsInRvVVRDU3RyaW5nIiwiVGFiUGFuZSIsIlRhYnMiLCJXb3Jrc3BhY2VzIiwid29ya3NwYWNlIiwic2V0V29ya3NwYWNlIiwid29ya3NwYWNlcyIsIm1vZGUiLCJvbmxpbmVXb3Jrc3BhY2UiLCJvZmZsaW5lV29yc2twYWNlIiwiaW5pdGlhbFdvcmtzcGFjZSIsIm9wZW5Xb3Jrc3BhY2VzIiwidG9nZ2xlV29ya3NwYWNlcyIsInN1YldvcmtzcGFjZSIsInNldFdvcmtzcGFjZVRvUXVlcnkiLCJIZWFkZXIiLCJpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCIsInNlbGVjdGVkRGF0YXNldFBhcnRzIiwiZ2V0RGF0YXNldE5hbWVTcGxpdEJ5U2xhc2hJbnRvT2JqZWN0IiwiZGF0YXNldFBhcnRzUG9zaXRpb25zIiwic29ydCIsImFsbERhdGFzZXRzIiwiZmlyc3RQb3NpdGlvbiIsImxhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb24iLCJwYXJ0c09iamVjdEFycmF5IiwiZ2V0RGF0YXNldFBhcnRzIiwiZGF0YXNldFBhcnRzIiwidmFsdWVzIiwidW5zaGlmdCIsImpvaW4iLCJpbmNsdWRlcyIsImdldEF2YWlsYWJsZUNob2ljZXMiLCJnZXRSZXN0T3B0aW9ucyIsIkluZGV4IiwiYmFja1RvTWFpblBhZ2UiLCJzdW1tYXJpZXNXb3Jrc3BhY2UiLCJmb2xkZXJzUGF0aCIsInRyaWdnZXJXb3Jrc3BhY2UiLCJ0cmFja2VyV29ya3NwYWNlIiwiY2Fsb3JpbWV0ZXJzV29ya3NwYWNlIiwibW91bnNXb3Jrc3BhY2UiLCJjdHRwc1dvcnNwYWNlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBRUE7QUFpQk8sSUFBTUEsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixPQUlIO0FBQUE7O0FBQUEsTUFIMUJDLGNBRzBCLFFBSDFCQSxjQUcwQjtBQUFBLE1BRjFCQyxLQUUwQixRQUYxQkEsS0FFMEI7QUFBQSxNQUQxQkMsZ0JBQzBCLFFBRDFCQSxnQkFDMEI7O0FBQUEsOEJBVXRCQywwSUFBNkMsQ0FDL0NELGdCQUQrQyxFQUUvQ0YsY0FGK0MsQ0FWdkI7QUFBQSxNQUV4Qkksb0NBRndCLHlCQUV4QkEsb0NBRndCO0FBQUEsTUFHeEJDLGdCQUh3Qix5QkFHeEJBLGdCQUh3QjtBQUFBLE1BSXhCQyxhQUp3Qix5QkFJeEJBLGFBSndCO0FBQUEsTUFLeEJDLCtCQUx3Qix5QkFLeEJBLCtCQUx3QjtBQUFBLE1BTXhCQyw0QkFOd0IseUJBTXhCQSw0QkFOd0I7QUFBQSxNQU94QkMsa0NBUHdCLHlCQU94QkEsa0NBUHdCO0FBQUEsTUFReEJDLDJDQVJ3Qix5QkFReEJBLDJDQVJ3QjtBQUFBLE1BU3hCQyxlQVR3Qix5QkFTeEJBLGVBVHdCOztBQWUxQkMseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBSUYsMkNBQUosRUFBaUQ7QUFDL0NHLG9GQUFZLENBQ1ZDLHVGQUFxQixDQUFDO0FBQUVDLG9CQUFZLEVBQUVKO0FBQWhCLE9BQUQsRUFBb0NWLEtBQXBDLENBRFgsQ0FBWjtBQUdEO0FBQ0YsR0FOUSxFQU1OLENBQUNVLGVBQUQsQ0FOTSxDQUFUO0FBUUEsU0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR1Asb0NBQW9DLENBQUNZLEdBQXJDLENBQXlDLFVBQUNDLElBQUQsRUFBZTtBQUN2RCxRQUFNQyxRQUFRLEdBQUdDLE1BQU0sQ0FBQ0MsSUFBUCxDQUFZSCxJQUFaLEVBQWtCLENBQWxCLENBQWpCO0FBQ0EsV0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLHlEQUFEO0FBQ0UsZUFBUyxFQUFFQSxJQUFJLENBQUNDLFFBQUQsQ0FBSixDQUFlRyxtQkFENUI7QUFFRSxVQUFJLEVBQUVILFFBRlI7QUFHRSxrQkFBWSxFQUFFRCxJQUFJLENBQUNDLFFBQUQsQ0FBSixDQUFlSSxnQkFIL0I7QUFJRSxnQkFBVSxFQUFFYixrQ0FKZDtBQUtFLGFBQU8sRUFBRUYsK0JBTFg7QUFNRSxrQkFBWSxFQUFFQyw0QkFOaEIsQ0FPRTtBQVBGO0FBUUUsVUFBSSxFQUFFRixhQUFhLENBQUNZLFFBQUQsQ0FSckI7QUFTRSxzQkFBZ0IsRUFBRWIsZ0JBVHBCO0FBVUUsbUJBQWEsRUFBRUMsYUFWakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGLENBREY7QUFnQkQsR0FsQkEsQ0FESCxFQW9CRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR0ksMkNBQTJDLEdBQzFDLE1BQUMsbUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUQwQyxHQUcxQyxNQUFDLGlFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFKSixDQXBCRixDQURGO0FBOEJELENBekRNOztHQUFNWCxlO1VBY1BJLGtJOzs7S0FkT0osZTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQ3hCYjtBQUNBO0FBRUE7QUFDQTtJQUNRd0IsTSxHQUFXQywyQyxDQUFYRCxNO0FBY0QsSUFBTUUsWUFBWSxHQUFHLFNBQWZBLFlBQWUsT0FVSDtBQUFBOztBQUFBLE1BVHZCQyxPQVN1QixRQVR2QkEsT0FTdUI7QUFBQSxNQVJ2QkMsVUFRdUIsUUFSdkJBLFVBUXVCO0FBQUEsTUFQdkJDLFlBT3VCLFFBUHZCQSxZQU91QjtBQUFBLE1BTnZCQyxTQU11QixRQU52QkEsU0FNdUI7QUFBQSxNQUx2QlosSUFLdUIsUUFMdkJBLElBS3VCO0FBQUEsTUFKdkJhLElBSXVCLFFBSnZCQSxJQUl1QjtBQUFBLE1BSHZCekIsZ0JBR3VCLFFBSHZCQSxnQkFHdUI7QUFBQSxNQUZ2QkMsYUFFdUIsUUFGdkJBLGFBRXVCO0FBQUEsTUFEdkJ5QixZQUN1QixRQUR2QkEsWUFDdUI7O0FBQUEsa0JBQ0dDLHNEQUFRLENBQUNGLElBQUQsQ0FEWDtBQUFBLE1BQ2hCRyxLQURnQjtBQUFBLE1BQ1RDLFFBRFM7O0FBQUEsbUJBRVNGLHNEQUFRLENBQUMsS0FBRCxDQUZqQjtBQUFBLE1BRWhCRyxVQUZnQjtBQUFBLE1BRUpDLFNBRkk7O0FBSXZCLFNBQ0UsTUFBQyw4RUFBRDtBQUNFLDRCQUF3QixFQUFFLEtBRDVCO0FBRUUsZ0JBQVksRUFBRU4sSUFGaEI7QUFHRSxZQUFRLEVBQUVDLFlBQVksS0FBS0UsS0FBakIsR0FBeUIsVUFBekIsR0FBc0MsRUFIbEQ7QUFJRSxZQUFRLEVBQUUsa0JBQUNBLEtBQUQsRUFBZ0I7QUFDeEIzQixtQkFBYSxDQUFDVyxJQUFELENBQWIsR0FBc0JnQixLQUF0QjtBQUNBNUIsc0JBQWdCLENBQUNDLGFBQUQsQ0FBaEI7QUFDQXFCLGdCQUFVLENBQUNWLElBQUQsQ0FBVjtBQUNBaUIsY0FBUSxDQUFDRCxLQUFELENBQVI7QUFDQVAsYUFBTyxDQUFDTyxLQUFELENBQVA7QUFDRCxLQVZIO0FBV0UsV0FBTyxFQUFFO0FBQUEsYUFBTUcsU0FBUyxDQUFDLENBQUNELFVBQUYsQ0FBZjtBQUFBLEtBWFg7QUFZRSxRQUFJLEVBQUVBLFVBWlI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQWNHUCxZQUFZLENBQUNaLEdBQWIsQ0FBaUIsVUFBQ3FCLE1BQUQ7QUFBQSxXQUNoQixNQUFDLE1BQUQ7QUFBUSxXQUFLLEVBQUVBLE1BQWY7QUFBdUIsU0FBRyxFQUFFQSxNQUE1QjtBQUFvQyxhQUFPLEVBQUU7QUFBQSxlQUFNRCxTQUFTLENBQUMsS0FBRCxDQUFmO0FBQUEsT0FBN0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNFLE1BQUMscUVBQUQ7QUFBcUIsa0JBQVksRUFBQyxXQUFsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0dDLE1BREgsQ0FERixDQURnQjtBQUFBLEdBQWpCLENBZEgsRUFxQkdSLFNBQVMsQ0FBQ2IsR0FBVixDQUFjLFVBQUNxQixNQUFEO0FBQUEsV0FDYixNQUFDLE1BQUQ7QUFBUSxTQUFHLEVBQUVBLE1BQWI7QUFBcUIsV0FBSyxFQUFFQSxNQUE1QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyxxRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQXNCQSxNQUF0QixDQURGLENBRGE7QUFBQSxHQUFkLENBckJILENBREY7QUE2QkQsQ0EzQ007O0dBQU1aLFk7O0tBQUFBLFk7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUNuQmI7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBR0E7QUFDQTtBQUtPLElBQU1hLE9BQU8sR0FBRyxTQUFWQSxPQUFVLEdBQU07QUFBQTs7QUFBQSxrQkFDZU4sc0RBQVEsQ0FDaERPLDREQUFpQixDQUFDLENBQUQsQ0FBakIsQ0FBcUJOLEtBRDJCLENBRHZCO0FBQUEsTUFDcEJPLGFBRG9CO0FBQUEsTUFDTEMsZ0JBREs7O0FBSTNCLE1BQU1DLE1BQU0sR0FBR0MsOERBQVMsRUFBeEI7QUFDQSxNQUFNMUMsS0FBaUIsR0FBR3lDLE1BQU0sQ0FBQ3pDLEtBQWpDO0FBRUEsTUFBTTJDLFVBQVUsR0FBRzNDLEtBQUssQ0FBQzJDLFVBQU4sR0FBbUIzQyxLQUFLLENBQUMyQyxVQUF6QixHQUFzQyxFQUF6RDtBQUNBLE1BQU03QixZQUFZLEdBQUdkLEtBQUssQ0FBQ2MsWUFBTixHQUFxQmQsS0FBSyxDQUFDYyxZQUEzQixHQUEwQyxFQUEvRDtBQUNBLE1BQU04QixJQUFJLEdBQUc1QyxLQUFLLENBQUM0QyxJQUFOLEdBQWFDLFFBQVEsQ0FBQzdDLEtBQUssQ0FBQzRDLElBQVAsQ0FBckIsR0FBb0NFLEdBQWpEOztBQVQyQiwwQkFXQUMsNENBQUssQ0FBQ0MsVUFBTixDQUFpQkMsZ0VBQWpCLENBWEE7QUFBQSxNQVduQkMsY0FYbUIscUJBV25CQSxjQVhtQjs7QUFBQSxtQkFZcUJuQixzREFBUSxDQUFDWSxVQUFELENBWjdCO0FBQUEsTUFZcEIxQyxnQkFab0I7QUFBQSxNQVlGa0QsbUJBWkU7O0FBQUEsbUJBYWlCcEIsc0RBQVEsQ0FBU2pCLFlBQVQsQ0FiekI7QUFBQSxNQWFwQmYsY0Fib0I7QUFBQSxNQWFKcUQsaUJBYkk7O0FBZTNCLE1BQU1DLHlCQUF5QixHQUFHLFNBQTVCQSx5QkFBNEIsQ0FBQ1QsSUFBRCxFQUFrQjtBQUNsRDtBQUNBaEMsbUZBQVksQ0FBQ0Msd0ZBQXFCLENBQUM7QUFBRStCLFVBQUksRUFBRUE7QUFBUixLQUFELEVBQWlCNUMsS0FBakIsQ0FBdEIsQ0FBWixDQUZrRCxDQUdsRDtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFDQWtELGtCQUFjLENBQUNOLElBQUQsQ0FBZDtBQUNELEdBWkQ7O0FBY0EsTUFBSTNDLGdCQUFnQixLQUFLRCxLQUFLLENBQUMyQyxVQUEzQixJQUF5QzVDLGNBQWMsS0FBS0MsS0FBSyxDQUFDYyxZQUF0RSxFQUFvRjtBQUNsRkYsbUZBQVksQ0FDVkMsd0ZBQXFCLENBQ25CO0FBQ0U4QixnQkFBVSxFQUFFMUMsZ0JBRGQ7QUFFRWEsa0JBQVksRUFBRWY7QUFGaEIsS0FEbUIsRUFLbkJDLEtBTG1CLENBRFgsQ0FBWjtBQVNELEdBdkMwQixDQXlDM0I7OztBQUNBLFNBQ0UsTUFBQyx5REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx1REFBRDtBQUFZLFNBQUssRUFBRUEsS0FBbkI7QUFBMEIsdUJBQW1CLEVBQUVtRCxtQkFBL0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsRUFJRSxNQUFDLCtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR0csK0RBQWdCLENBQUNDLFlBQWpCLENBQThCQyxlQUE5QixJQUNDLE1BQUMsdUVBQUQ7QUFDRSxzQkFBa0IsRUFBRVosSUFEdEI7QUFFRSxvQkFBZ0IsRUFBRTNDLGdCQUZwQjtBQUdFLGtCQUFjLEVBQUVGLGNBSGxCO0FBSUUsV0FBTyxFQUFFc0QseUJBSlg7QUFLRSxTQUFLLEVBQUMsT0FMUjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRkosQ0FKRixFQWVFLE1BQUMsZ0VBQUQ7QUFDRSxjQUFVLEVBQUMsT0FEYjtBQUVFLFNBQUssRUFDSCxNQUFDLG1EQUFEO0FBQ0UsYUFBTyxFQUFFZiw0REFEWDtBQUVFLFlBQU0sRUFBRUUsZ0JBRlY7QUFHRSxrQkFBWSxFQUFFRiw0REFBaUIsQ0FBQyxDQUFELENBSGpDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFISjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBVUdDLGFBQWEsS0FBS0QsNERBQWlCLENBQUMsQ0FBRCxDQUFqQixDQUFxQk4sS0FBdkMsR0FDQyxNQUFDLCtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGlGQUFEO0FBQ0UscUJBQWlCLEVBQUVvQixpQkFEckI7QUFFRSxTQUFLLEVBQUVwRCxLQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURELEdBUUcsTUFBQywrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxvRkFBRDtBQUNFLG9CQUFnQixFQUFFQyxnQkFEcEI7QUFFRSxrQkFBYyxFQUFFRixjQUZsQjtBQUdFLFNBQUssRUFBRUMsS0FIVDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FsQk4sQ0FmRixDQURGLENBREY7QUErQ0QsQ0F6Rk07O0dBQU1xQyxPO1VBSUlLLHNEOzs7S0FKSkwsTzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDckJiO0FBQ0E7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUlBO0lBR1FmLE0sR0FBV0MsMkMsQ0FBWEQsTTtBQWVELElBQU1tQyxrQkFBa0IsR0FBRyxTQUFyQkEsa0JBQXFCLE9BTUg7QUFBQTs7QUFBQSxNQUw3QkMsS0FLNkIsUUFMN0JBLEtBSzZCO0FBQUEsTUFKN0JDLGtCQUk2QixRQUo3QkEsa0JBSTZCO0FBQUEsTUFIN0JDLE9BRzZCLFFBSDdCQSxPQUc2QjtBQUFBLE1BRjdCM0QsZ0JBRTZCLFFBRjdCQSxnQkFFNkI7QUFBQSxNQUQ3QkYsY0FDNkIsUUFEN0JBLGNBQzZCOztBQUM3QjtBQUNBO0FBRjZCLHdCQUdXZ0QsOENBQUEsQ0FBZSxDQUNyRDtBQUFFYyxTQUFLLEVBQUUsS0FBVDtBQUFnQjdCLFNBQUssRUFBRTtBQUF2QixHQURxRCxDQUFmLENBSFg7QUFBQTtBQUFBLE1BR3RCOEIsWUFIc0I7QUFBQSxNQUdSQyxlQUhROztBQU83QixNQUFNQyxZQUFZLEdBQUcsSUFBSUMsSUFBSixHQUFXQyxPQUFYLEVBQXJCOztBQVA2Qix5QkFRZ0JuQiw4Q0FBQSxDQUFlaUIsWUFBZixDQVJoQjtBQUFBO0FBQUEsTUFRdEJHLGNBUnNCO0FBQUEsTUFRTkMsa0JBUk0sd0JBVTdCOzs7QUFWNkIsb0JBV09DLG9FQUFVLENBQzVDQyxzRUFBZSxDQUFDO0FBQ2QzQixjQUFVLEVBQUUxQyxnQkFERTtBQUVkYSxnQkFBWSxFQUFFZixjQUZBO0FBR2Q2QyxRQUFJLEVBQUUsQ0FBQyxDQUhPO0FBSWQyQixnQkFBWSxFQUFFSjtBQUpBLEdBQUQsQ0FENkIsRUFPNUMsRUFQNEMsRUFRNUMsQ0FBQ2xFLGdCQUFELEVBQW1CRixjQUFuQixFQUFtQ29FLGNBQW5DLENBUjRDLENBWGpCO0FBQUEsTUFXckJLLElBWHFCLGVBV3JCQSxJQVhxQjtBQUFBLE1BV2ZDLFNBWGUsZUFXZkEsU0FYZTtBQUFBLE1BV0pDLE1BWEksZUFXSkEsTUFYSTs7QUFxQjdCLE1BQU1DLGtCQUFrQixHQUFHSCxJQUFJLEdBQUdBLElBQUksQ0FBQ0EsSUFBUixHQUFlLEVBQTlDO0FBRUF6QixpREFBQSxDQUFnQixZQUFNO0FBQ3BCO0FBQ0EsUUFBTTZCLHFCQUFvQyxHQUN4Q0Qsa0JBQWtCLENBQUNFLE1BQW5CLEdBQTRCLENBQTVCLEdBQ0lGLGtCQUFrQixDQUFDNUQsR0FBbkIsQ0FBdUIsVUFBQytELEdBQUQsRUFBK0I7QUFDcEQsYUFBTztBQUFFakIsYUFBSyxFQUFFaUIsR0FBRyxDQUFDbEMsSUFBSixDQUFTbUMsUUFBVCxFQUFUO0FBQThCL0MsYUFBSyxFQUFFOEMsR0FBRyxDQUFDbEM7QUFBekMsT0FBUDtBQUNELEtBRkQsQ0FESixHQUlJLEVBTE47O0FBTUEsUUFBTW9DLElBQUksR0FBRyw2RkFBSWxCLFlBQVAsQ0FBVjs7QUFDQSxRQUFNbUIsUUFBUSxHQUFHRCxJQUFJLENBQUNFLE1BQUwsQ0FBWU4scUJBQVosQ0FBakI7QUFDQWIsbUJBQWUsQ0FBQ2tCLFFBQUQsQ0FBZjtBQUNELEdBWEQsRUFXRyxDQUFDTixrQkFBRCxDQVhIO0FBYUEsTUFBTVEsVUFBVSxHQUFHckIsWUFBWSxDQUFDL0MsR0FBYixDQUFpQixVQUFDNkIsSUFBRDtBQUFBLFdBQXVCQSxJQUFJLENBQUNaLEtBQTVCO0FBQUEsR0FBakIsQ0FBbkIsQ0FwQzZCLENBc0M3QjtBQUNBO0FBQ0E7O0FBQ0EsTUFBTW9ELGdCQUFnQixHQUNwQkQsVUFBVSxDQUFDRSxPQUFYLENBQW1CMUIsa0JBQW5CLE1BQTJDLENBQUMsQ0FBNUMsR0FDSSxDQURKLEdBRUl3QixVQUFVLENBQUNFLE9BQVgsQ0FBbUIxQixrQkFBbkIsQ0FITjtBQUlBLFNBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixjQUFVLEVBQUVELEtBQTVCO0FBQW1DLFFBQUksRUFBRSxNQUF6QztBQUFpRCxTQUFLLEVBQUMsTUFBdkQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBSyxXQUFPLEVBQUMsUUFBYjtBQUFzQixTQUFLLEVBQUMsUUFBNUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFDRSxZQUFRLEVBQUUsQ0FBQ0ksWUFBWSxDQUFDc0IsZ0JBQWdCLEdBQUcsQ0FBcEIsQ0FEekI7QUFFRSxRQUFJLEVBQUUsTUFBQyxpRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BRlI7QUFHRSxRQUFJLEVBQUMsTUFIUDtBQUlFLFdBQU8sRUFBRSxtQkFBTTtBQUNieEIsYUFBTyxDQUFDdUIsVUFBVSxDQUFDQyxnQkFBZ0IsR0FBRyxDQUFwQixDQUFYLENBQVA7QUFDRCxLQU5IO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLEVBWUUsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw4RUFBRDtBQUNFLDRCQUF3QixFQUFFLEtBRDVCO0FBRUUsU0FBSyxFQUFFRCxVQUFVLENBQUNDLGdCQUFELENBRm5CO0FBR0UsWUFBUSxFQUFFLGtCQUFDRSxDQUFELEVBQVk7QUFDcEIxQixhQUFPLENBQUNmLFFBQVEsQ0FBQ3lDLENBQUQsQ0FBVCxDQUFQO0FBQ0QsS0FMSDtBQU1FLGNBQVUsRUFBRSxJQU5kO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FRR3hCLFlBQVksSUFDWEEsWUFBWSxDQUFDL0MsR0FBYixDQUFpQixVQUFDd0UsbUJBQUQsRUFBc0M7QUFDckQsV0FDRSxNQUFDLE1BQUQ7QUFDRSxXQUFLLEVBQUVBLG1CQUFtQixDQUFDdkQsS0FEN0I7QUFFRSxTQUFHLEVBQUV1RCxtQkFBbUIsQ0FBQ3ZELEtBQXBCLENBQTBCK0MsUUFBMUIsRUFGUDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BSUdOLFNBQVMsR0FDUixNQUFDLGlGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERixDQURRLEdBS1I7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFJYyxtQkFBbUIsQ0FBQzFCLEtBQXhCLENBVEosQ0FERjtBQWNELEdBZkQsQ0FUSixDQURGLENBWkYsRUF3Q0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUNFLFFBQUksRUFBRSxNQUFDLGtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEUjtBQUVFLFlBQVEsRUFBRSxDQUFDQyxZQUFZLENBQUNzQixnQkFBZ0IsR0FBRyxDQUFwQixDQUZ6QjtBQUdFLFFBQUksRUFBQyxNQUhQO0FBSUUsV0FBTyxFQUFFLG1CQUFNO0FBQ2J4QixhQUFPLENBQUN1QixVQUFVLENBQUNDLGdCQUFnQixHQUFHLENBQXBCLENBQVgsQ0FBUDtBQUNELEtBTkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBeENGLENBREYsQ0FERixDQURGO0FBeURELENBNUdNOztHQUFNM0Isa0I7VUFpQnlCWSw0RDs7O0tBakJ6Qlosa0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQzVCYjtBQUNBO0FBQ0E7QUFVTyxJQUFNK0IsWUFBWSxHQUFHLFNBQWZBLFlBQWUsT0FBa0Q7QUFBQTs7QUFBQSxNQUEvQ0MsT0FBK0MsUUFBL0NBLE9BQStDO0FBQUEsTUFBdENDLFlBQXNDLFFBQXRDQSxZQUFzQztBQUFBLE1BQXhCQyxNQUF3QixRQUF4QkEsTUFBd0I7O0FBQUEsa0JBQ2xENUQsc0RBQVEsQ0FBQzJELFlBQUQsQ0FEMEM7QUFBQSxNQUNyRTFELEtBRHFFO0FBQUEsTUFDOURDLFFBRDhEOztBQUU1RSxNQUFNMkQsUUFBUSxHQUFHLFNBQVhBLFFBQVcsQ0FBQ0gsT0FBRCxFQUF5QkMsWUFBekI7QUFBQSxXQUNmLE1BQUMseUNBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHRCxPQUFPLENBQUMxRSxHQUFSLENBQVksVUFBQzhFLE1BQUQ7QUFBQSxhQUNYLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsV0FBRyxFQUFFQSxNQUFNLENBQUM3RCxLQURkO0FBRUUsZUFBTyxFQUFFLG1CQUFNO0FBQ2IyRCxnQkFBTSxJQUFJQSxNQUFNLENBQUNFLE1BQU0sQ0FBQzdELEtBQVIsQ0FBaEI7QUFDQUMsa0JBQVEsQ0FBQzRELE1BQUQsQ0FBUjtBQUNELFNBTEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQU9FO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FBTUEsTUFBTSxDQUFDaEMsS0FBYixDQVBGLENBRFc7QUFBQSxLQUFaLENBREgsQ0FEZTtBQUFBLEdBQWpCOztBQWdCQSxTQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNkNBQUQ7QUFBVSxXQUFPLEVBQUUrQixRQUFRLENBQUNILE9BQUQsRUFBVUMsWUFBVixDQUEzQjtBQUFvRCxXQUFPLEVBQUUsQ0FBQyxPQUFELENBQTdEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRTtBQUFHLFNBQUssRUFBRTtBQUFFaEMsV0FBSyxFQUFFO0FBQVQsS0FBVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0cxQixLQUFLLENBQUM2QixLQURULE9BQ2dCLE1BQUMsOERBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURoQixFQUNpQyxHQURqQyxDQURGLENBREYsQ0FERixDQURGO0FBV0QsQ0E3Qk07O0dBQU0yQixZOztLQUFBQSxZOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDWmI7QUFDQTtBQUVBO0FBQ0E7QUFFQTtBQUNBO0FBRU8sSUFBTU0saUJBQWlCLEdBQUcsU0FBcEJBLGlCQUFvQixHQUFNO0FBQUE7O0FBQ3JDLE1BQU1yRCxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTTFDLEtBQWlCLEdBQUd5QyxNQUFNLENBQUN6QyxLQUFqQztBQUVBLE1BQU04RSxHQUFHLEdBQUc5RSxLQUFLLENBQUMyQyxVQUFOLEdBQW1CM0MsS0FBSyxDQUFDMkMsVUFBekIsR0FBc0MsRUFBbEQ7O0FBSnFDLHdCQU1XSSw4Q0FBQSxDQUFlK0IsR0FBZixDQU5YO0FBQUE7QUFBQSxNQU05QmlCLGlCQU44QjtBQUFBLE1BTVhDLGtCQU5XOztBQUFBLHlCQU9lakQsOENBQUEsQ0FDbEQvQyxLQUFLLENBQUNjLFlBRDRDLENBUGY7QUFBQTtBQUFBLE1BTzlCbUYsbUJBUDhCO0FBQUEsTUFPVEMsb0JBUFM7O0FBQUEseUJBVURuRCw4Q0FBQSxDQUFlLEtBQWYsQ0FWQztBQUFBO0FBQUEsTUFVOUJvRCxVQVY4QjtBQUFBLE1BVWxCQyxhQVZrQjs7QUFZckNyRCxpREFBQSxDQUFnQixZQUFNO0FBQ3BCO0FBQ0EsUUFBSW9ELFVBQUosRUFBZ0I7QUFDZCxVQUFNckIsSUFBRyxHQUFHOUUsS0FBSyxDQUFDMkMsVUFBTixHQUFtQjNDLEtBQUssQ0FBQzJDLFVBQXpCLEdBQXNDLEVBQWxEOztBQUNBdUQsMEJBQW9CLENBQUNsRyxLQUFLLENBQUNjLFlBQVAsQ0FBcEI7QUFDQWtGLHdCQUFrQixDQUFDbEIsSUFBRCxDQUFsQjtBQUNEO0FBQ0YsR0FQRCxFQU9HLENBQUNxQixVQUFELENBUEg7QUFTQSxTQUNFLE1BQUMsMkRBQUQ7QUFBVyxXQUFPLEVBQUMsTUFBbkI7QUFBMEIsY0FBVSxFQUFDLFFBQXJDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGtFQUFEO0FBQ0UsY0FBVSxFQUFFQSxVQURkO0FBRUUsaUJBQWEsRUFBRUMsYUFGakI7QUFHRSxzQkFBa0IsRUFBRUosa0JBSHRCO0FBSUUsd0JBQW9CLEVBQUVFLG9CQUp4QjtBQUtFLHFCQUFpQixFQUFFSCxpQkFMckI7QUFNRSx1QkFBbUIsRUFBRUUsbUJBTnZCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixFQVNFLE1BQUMsMkRBQUQ7QUFBVyxTQUFLLEVBQUMsYUFBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsaURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBRUUsTUFBQywwREFBRDtBQUFjLFdBQU8sRUFBRTtBQUFBLGFBQU1HLGFBQWEsQ0FBQyxJQUFELENBQW5CO0FBQUEsS0FBdkI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUZGLENBVEYsQ0FERjtBQWdCRCxDQXJDTTs7R0FBTU4saUI7VUFDSXBELHFEOzs7S0FESm9ELGlCOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDVGI7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBRU8sSUFBTU8sY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixHQUFNO0FBQUE7O0FBQ2xDLE1BQU01RCxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTTFDLEtBQWlCLEdBQUd5QyxNQUFNLENBQUN6QyxLQUFqQztBQUVBLE1BQU1zRyxnQkFBZ0IsR0FDcEJ0RyxLQUFLLENBQUMyQyxVQUFOLEtBQXFCLEdBQXJCLElBQTRCM0MsS0FBSyxDQUFDYyxZQUFOLEtBQXVCLG9CQURyRDtBQUdBLFNBQ0UsTUFBQywyREFBRDtBQUNFLFNBQUssRUFBQyxNQURSO0FBRUUsV0FBTyxFQUFDLE1BRlY7QUFHRSxrQkFBYyxFQUFDLGVBSGpCO0FBSUUsY0FBVSxFQUFDLFFBSmI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU1Hd0YsZ0JBQWdCLEdBQ2YsTUFBQyw4REFBRDtBQUFnQixTQUFLLEVBQUV0RyxLQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRGUsR0FHZixNQUFDLHNFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFUSixFQVdFLE1BQUMsK0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsbURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsRUFJRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGlFQUFEO0FBQVksb0JBQWdCLEVBQUUsS0FBOUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBSkYsQ0FYRixDQURGO0FBc0JELENBN0JNOztHQUFNcUcsYztVQUNJM0QscUQ7OztLQURKMkQsYzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQ1piO0FBQ0E7QUFDQTtBQUNBO0FBRUE7QUFJQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUFZQSxJQUFNRSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLENBQUN2RyxLQUFELEVBQW1CO0FBQ3hDd0csUUFBTSxDQUFDQyxJQUFQLENBQVl6RyxLQUFaLEVBQW1CLFFBQW5CO0FBQ0QsQ0FGRDs7QUFJTyxJQUFNMEcsV0FBVyxHQUFHLFNBQWRBLFdBQWMsT0FPQztBQUFBOztBQUFBLE1BTjFCTixhQU0wQixRQU4xQkEsYUFNMEI7QUFBQSxNQUwxQkQsVUFLMEIsUUFMMUJBLFVBSzBCO0FBQUEsTUFKMUJKLGlCQUkwQixRQUoxQkEsaUJBSTBCO0FBQUEsTUFIMUJFLG1CQUcwQixRQUgxQkEsbUJBRzBCO0FBQUEsTUFGMUJDLG9CQUUwQixRQUYxQkEsb0JBRTBCO0FBQUEsTUFEMUJGLGtCQUMwQixRQUQxQkEsa0JBQzBCO0FBQzFCLE1BQU12RCxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTTFDLEtBQWlCLEdBQUd5QyxNQUFNLENBQUN6QyxLQUFqQztBQUNBLE1BQU0yRyxPQUFPLEdBQUczRyxLQUFLLENBQUNjLFlBQU4sR0FBcUJkLEtBQUssQ0FBQ2MsWUFBM0IsR0FBMEMsRUFBMUQ7O0FBSDBCLGtCQUtZaUIsc0RBQVEsQ0FBQzRFLE9BQUQsQ0FMcEI7QUFBQSxNQUtuQkMsV0FMbUI7QUFBQSxNQUtOQyxjQUxNOztBQUFBLG1CQU1tQjlFLHNEQUFRLENBQUMsS0FBRCxDQU4zQjtBQUFBLE1BTW5CK0UsZUFObUI7QUFBQSxNQU1GQyxpQkFORTs7QUFPMUIsTUFBTWpDLEdBQUcsR0FBRzlFLEtBQUssQ0FBQzJDLFVBQU4sR0FBbUIzQyxLQUFLLENBQUMyQyxVQUF6QixHQUFzQyxFQUFsRDs7QUFQMEIsbUJBUVFaLHNEQUFRLENBQVMrQyxHQUFULENBUmhCO0FBQUEsTUFRbkJrQyxTQVJtQjtBQUFBLE1BUVJDLFlBUlE7O0FBVTFCdEcseURBQVMsQ0FBQyxZQUFNO0FBQ2QsUUFBTW1FLEdBQUcsR0FBRzlFLEtBQUssQ0FBQzJDLFVBQU4sR0FBbUIzQyxLQUFLLENBQUMyQyxVQUF6QixHQUFzQyxFQUFsRDtBQUNBLFFBQU1nRSxPQUFPLEdBQUczRyxLQUFLLENBQUNjLFlBQU4sR0FBcUJkLEtBQUssQ0FBQ2MsWUFBM0IsR0FBMEMsRUFBMUQ7QUFDQStGLGtCQUFjLENBQUNGLE9BQUQsQ0FBZDtBQUNBTSxnQkFBWSxDQUFDbkMsR0FBRCxDQUFaO0FBQ0QsR0FMUSxFQUtOLENBQUM5RSxLQUFLLENBQUNjLFlBQVAsRUFBcUJkLEtBQUssQ0FBQzJDLFVBQTNCLENBTE0sQ0FBVDs7QUFPQSxNQUFNdUUsU0FBUyxHQUFHLFNBQVpBLFNBQVksR0FBTTtBQUN0QmQsaUJBQWEsQ0FBQyxLQUFELENBQWI7QUFDRCxHQUZEOztBQUlBLE1BQU1lLGFBQWEsR0FBRyxTQUFoQkEsYUFBZ0IsQ0FBQ3hFLFVBQUQsRUFBcUI3QixZQUFyQixFQUE4QztBQUNsRStGLGtCQUFjLENBQUMvRixZQUFELENBQWQ7QUFDQW1HLGdCQUFZLENBQUN0RSxVQUFELENBQVo7QUFDRCxHQUhEOztBQUtBLE1BQU15RSxpQkFBaUIsR0FBRyxTQUFwQkEsaUJBQW9CLENBQ3hCQyxvQkFEd0IsRUFFeEJDLHNCQUZ3QixFQUdyQjtBQUNIdEIsc0JBQWtCLENBQUNxQixvQkFBRCxDQUFsQjtBQUNBbkIsd0JBQW9CLENBQUNvQixzQkFBRCxDQUFwQjtBQUNELEdBTkQ7O0FBMUIwQixtQkFrQ2dDQyxrRUFBUyxDQUNqRXhCLGlCQURpRSxFQUVqRUUsbUJBRmlFLENBbEN6QztBQUFBLE1Ba0NsQnVCLGVBbENrQixjQWtDbEJBLGVBbENrQjtBQUFBLE1Ba0NEQyxTQWxDQyxjQWtDREEsU0FsQ0M7QUFBQSxNQWtDVWhELFNBbENWLGNBa0NVQSxTQWxDVjtBQUFBLE1Ba0NxQkMsTUFsQ3JCLGNBa0NxQkEsTUFsQ3JCOztBQXVDMUIsTUFBTWdELElBQUk7QUFBQSxpTUFBRztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxtQkFDUFosZUFETztBQUFBO0FBQUE7QUFBQTs7QUFFSGEsb0JBRkcsR0FFTUMsSUFBSSxDQUFDQyxjQUFMLEVBRk47QUFHSEMsa0NBSEcsR0FHb0JDLHlDQUFFLENBQUNDLFNBQUgsQ0FDM0JuSCx3RkFBcUIsQ0FBQzhHLE1BQUQsRUFBUzNILEtBQVQsQ0FETSxDQUhwQixFQU1UO0FBQ0E7QUFDQTs7QUFDTWlJLDBCQVRHLEdBU1l6QixNQUFNLENBQUMwQixRQUFQLENBQWdCQyxJQUFoQixDQUFxQkMsS0FBckIsQ0FBMkIsSUFBM0IsRUFBaUMsQ0FBakMsQ0FUWjtBQVVUN0IsNEJBQWMsV0FBSTBCLFlBQUosZUFBcUJILG9CQUFyQixFQUFkO0FBVlM7QUFBQTs7QUFBQTtBQUFBO0FBQUEscUJBWUhGLElBQUksQ0FBQ1MsTUFBTCxFQVpHOztBQUFBO0FBY1huQix1QkFBUzs7QUFkRTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUFIOztBQUFBLG9CQUFKUSxJQUFJO0FBQUE7QUFBQTtBQUFBLEtBQVY7O0FBdkMwQixzQkF3RFhZLHlDQUFJLENBQUNDLE9BQUwsRUF4RFc7QUFBQTtBQUFBLE1Bd0RuQlgsSUF4RG1COztBQTBEMUIsU0FDRSxNQUFDLDZFQUFEO0FBQ0UsU0FBSyxFQUFDLGFBRFI7QUFFRSxXQUFPLEVBQUV6QixVQUZYO0FBR0UsWUFBUSxFQUFFO0FBQUEsYUFBTWUsU0FBUyxFQUFmO0FBQUEsS0FIWjtBQUlFLFVBQU0sRUFBRSxDQUNOLE1BQUMsK0RBQUQ7QUFDRSxXQUFLLEVBQUVzQixvREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRGhDO0FBRUUsZ0JBQVUsRUFBQyxPQUZiO0FBR0UsU0FBRyxFQUFDLE9BSE47QUFJRSxhQUFPLEVBQUU7QUFBQSxlQUFNekIsU0FBUyxFQUFmO0FBQUEsT0FKWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGVBRE0sRUFTTixNQUFDLCtEQUFEO0FBQWMsU0FBRyxFQUFDLElBQWxCO0FBQXVCLGFBQU8sRUFBRVEsSUFBaEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxZQVRNLENBSlY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQWtCR3ZCLFVBQVUsSUFDVCxtRUFDRSxNQUFDLDZDQUFEO0FBQ0UsNkJBQXlCLEVBQUVKLGlCQUQ3QjtBQUVFLCtCQUEyQixFQUFFRSxtQkFGL0I7QUFHRSxzQkFBa0IsRUFBRVcsV0FIdEI7QUFJRSxvQkFBZ0IsRUFBRUksU0FKcEI7QUFLRSxXQUFPLEVBQUVJLGlCQUxYO0FBTUUsUUFBSSxFQUFDLEtBTlA7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBU0UsTUFBQywyREFBRDtBQUNFLFFBQUksRUFBRVEsSUFEUjtBQUVFLGdCQUFZLEVBQUVoQixXQUZoQjtBQUdFLGNBQVUsRUFBRUksU0FIZDtBQUlFLHFCQUFpQixFQUFFRCxpQkFKckI7QUFLRSxtQkFBZSxFQUFFRCxlQUxuQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBVEYsRUFnQkdXLFNBQVMsR0FDUixNQUFDLGdGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdFQUFEO0FBQ0UsV0FBTyxFQUFFTixhQURYO0FBRUUsYUFBUyxFQUFFMUMsU0FGYjtBQUdFLG1CQUFlLEVBQUUrQyxlQUhuQjtBQUlFLFVBQU0sRUFBRTlDLE1BSlY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBRFEsR0FVUixNQUFDLGdGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUExQkosQ0FuQkosQ0FERjtBQW9ERCxDQXJITTs7R0FBTWdDLFc7VUFRSWhFLHFELEVBaUMyQzZFLDBELEVBc0IzQ2UseUNBQUksQ0FBQ0MsTzs7O0tBL0RUN0IsVzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDaENiO0FBQ0E7QUFDQTtBQUVBO0FBTUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtJQUNRa0MsSyxHQUFVQywrQyxDQUFWRCxLO0FBTUQsSUFBTUUsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUFvQztBQUFBOztBQUFBOztBQUFBLE1BQWpDOUksS0FBaUMsUUFBakNBLEtBQWlDOztBQUFBLDJCQUNqQytJLG9GQUFpQixFQURnQjtBQUFBLE1BQ3hEQyxNQUR3RCxzQkFDeERBLE1BRHdEO0FBQUEsTUFDaERDLFVBRGdELHNCQUNoREEsVUFEZ0Q7O0FBRWhFLE1BQU1DLFdBQVcsR0FBR25HLGdEQUFBLENBQWlCRSwrREFBakIsQ0FBcEI7QUFDQSxTQUNFLDREQUNFLE1BQUMsNERBQUQ7QUFBWSxXQUFPLEVBQUMsTUFBcEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHa0csd0RBQWEsQ0FBQ3BJLEdBQWQsSUFBa0IsVUFBQ3FJLElBQUQsRUFBcUI7QUFBQTs7QUFDdEMsUUFBTUMsY0FBYyxHQUFHQyx1RkFBa0IsQ0FDdkNKLFdBRHVDLEVBRXZDbEosS0FGdUMsRUFHdkNvSixJQUFJLENBQUNwSCxLQUhrQyxFQUl2QyxnQkFKdUMsQ0FBekM7O0FBRHNDLHNCQVFWcUMsb0VBQVUsQ0FDcENrRixzRUFBYyxDQUFDRixjQUFELENBRHNCLEVBRXBDLEVBRm9DLEVBR3BDLENBQUNySixLQUFLLENBQUNjLFlBQVAsRUFBcUJkLEtBQUssQ0FBQzJDLFVBQTNCLENBSG9DLENBUkE7QUFBQSxRQVE5QjZCLElBUjhCLGVBUTlCQSxJQVI4QjtBQUFBLFFBUXhCQyxTQVJ3QixlQVF4QkEsU0FSd0I7O0FBYXRDLFdBQ0UsTUFBQywrREFBRDtBQUNFLFdBQUssRUFBQyxHQURSO0FBRUUsV0FBSyxFQUFDLGFBRlI7QUFHRSxXQUFLLEVBQUUrRCxtREFBSyxDQUFDQyxNQUFOLENBQWFlLE1BQWIsQ0FBb0JDLEtBSDdCO0FBSUUsVUFBSSxFQUFFTCxJQUFJLENBQUN2RixLQUpiO0FBS0UsV0FBSyxFQUFFdUYsSUFBSSxDQUFDdkYsS0FMZDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BT0UsTUFBQyxLQUFEO0FBQ0UsV0FBSyxFQUFFLENBRFQ7QUFFRSxXQUFLLEVBQUU7QUFDTEgsYUFBSyxZQUNIc0YsTUFBTSxHQUNGUixtREFBSyxDQUFDQyxNQUFOLENBQWFpQixZQUFiLENBQTBCQyxPQUR4QixHQUVGbkIsbURBQUssQ0FBQ0MsTUFBTixDQUFhaUIsWUFBYixDQUEwQkUsS0FIM0I7QUFEQSxPQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FVR25GLFNBQVMsR0FBRyxNQUFDLHlDQUFEO0FBQU0sVUFBSSxFQUFDLE9BQVg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUFILEdBQTJCb0YseURBQVMsQ0FBQ1QsSUFBRCxFQUFPNUUsSUFBUCxDQVZoRCxDQVBGLENBREY7QUFzQkQsR0FuQ0E7QUFBQSxZQVE2QkgsNERBUjdCO0FBQUEsS0FESCxDQURGLEVBdUNFLE1BQUMsMkRBQUQ7QUFDRSxrQkFBYyxFQUFDLFVBRGpCO0FBRUUsV0FBTyxFQUFDLE1BRlY7QUFHRSxjQUFVLEVBQUMsUUFIYjtBQUlFLGlCQUFhLEVBQUMsV0FKaEI7QUFLRSxTQUFLLEVBQ0gyRSxNQUFNLEdBQ0ZSLG1EQUFLLENBQUNDLE1BQU4sQ0FBYWlCLFlBQWIsQ0FBMEJDLE9BRHhCLEdBRUZuQixtREFBSyxDQUFDQyxNQUFOLENBQWFpQixZQUFiLENBQTBCRSxLQVJsQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGtCQVlFLE1BQUMsMkRBQUQ7QUFBVyxTQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNENBQUQ7QUFBUyxTQUFLLDZCQUFzQlosTUFBTSxHQUFHLElBQUgsR0FBVSxLQUF0QyxDQUFkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsUUFBSSxFQUFDLFNBRFA7QUFFRSxTQUFLLEVBQUMsUUFGUjtBQUdFLFdBQU8sRUFBRSxtQkFBTTtBQUNiQyxnQkFBVSxDQUFDLENBQUNELE1BQUYsQ0FBVjtBQUNELEtBTEg7QUFNRSxRQUFJLEVBQUVBLE1BQU0sR0FBRyxNQUFDLCtEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFBSCxHQUF1QixNQUFDLG9FQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFOckM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsQ0FaRixDQXZDRixDQURGO0FBbUVELENBdEVNOztJQUFNRixjO1VBQ29CQyw0RTs7O0tBRHBCRCxjOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUN6QmI7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUlBO0FBRUE7QUFXQSxJQUFNZ0IseUJBQXlCLEdBQUcsQ0FDaEM7QUFDRTlILE9BQUssRUFBRSxJQURUO0FBRUU2QixPQUFLLEVBQUU7QUFGVCxDQURnQyxFQUtoQztBQUNFN0IsT0FBSyxFQUFFLEtBRFQ7QUFFRTZCLE9BQUssRUFBRTtBQUZULENBTGdDLENBQWxDO0FBV08sSUFBTWtHLFlBQVksR0FBRyxTQUFmQSxZQUFlLE9BT0g7QUFBQTs7QUFBQSxNQUx2QmpKLFlBS3VCLFFBTHZCQSxZQUt1QjtBQUFBLE1BSnZCNkIsVUFJdUIsUUFKdkJBLFVBSXVCO0FBQUEsTUFIdkJpRixJQUd1QixRQUh2QkEsSUFHdUI7QUFBQSxNQUZ2QmIsaUJBRXVCLFFBRnZCQSxpQkFFdUI7QUFBQSxNQUR2QkQsZUFDdUIsUUFEdkJBLGVBQ3VCOztBQUFBLDBCQUNpQi9ELGdEQUFBLENBQWlCRSwrREFBakIsQ0FEakI7QUFBQSxNQUNmK0csV0FEZSxxQkFDZkEsV0FEZTtBQUFBLE1BQ0Y5RyxjQURFLHFCQUNGQSxjQURFOztBQUV2QixNQUFNVCxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTTFDLEtBQWlCLEdBQUd5QyxNQUFNLENBQUN6QyxLQUFqQzs7QUFFQSxNQUFNcUQseUJBQXlCLEdBQUcsU0FBNUJBLHlCQUE0QixDQUFDVCxJQUFELEVBQWtCO0FBQ2xEO0FBQ0E7QUFDQTtBQUNBTSxrQkFBYyxDQUFDTixJQUFELENBQWQ7QUFDRCxHQUxEOztBQU9BLFNBQ0UsTUFBQyx5REFBRDtBQUNFLFFBQUksRUFBRWdGLElBRFI7QUFFRSxZQUFRLEVBQUUsa0JBQUNELE1BQUQsRUFBWTtBQUNwQjtBQUNBL0csb0ZBQVksQ0FBQ0MsdUZBQXFCLENBQUM4RyxNQUFELEVBQVMzSCxLQUFULENBQXRCLENBQVo7QUFDRCxLQUxIO0FBTUUsVUFBTSxFQUFFLENBQ047QUFBRTZCLFVBQUksRUFBRSxjQUFSO0FBQXdCRyxXQUFLLEVBQUVsQjtBQUEvQixLQURNLEVBRU47QUFBRWUsVUFBSSxFQUFFLFlBQVI7QUFBc0JHLFdBQUssRUFBRVc7QUFBN0IsS0FGTSxFQUdOO0FBQUVkLFVBQUksRUFBRSxNQUFSO0FBQWdCRyxXQUFLLEVBQUVnSTtBQUF2QixLQUhNLENBTlY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQVlFO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFaRixFQWFFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0VBQUQ7QUFBZ0IsUUFBSSxFQUFFLGNBQXRCO0FBQXNDLFNBQUssRUFBQyxjQUE1QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxpRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQWtCbEosWUFBbEIsQ0FERixDQURGLENBYkYsRUFrQkUsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixRQUFJLEVBQUUsWUFBdEI7QUFBb0MsU0FBSyxFQUFDLFlBQTFDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGlFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FBa0I2QixVQUFsQixDQURGLENBREYsQ0FsQkYsRUF1QkdXLCtEQUFnQixDQUFDQyxZQUFqQixDQUE4QkMsZUFBOUIsSUFDQyxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGdGQUFEO0FBQ0UsU0FBSyxFQUFDLE9BRFI7QUFFRSxXQUFPLEVBQUVILHlCQUZYO0FBR0Usc0JBQWtCLEVBQUUyRyxXQUh0QjtBQUlFLGtCQUFjLEVBQUVsSixZQUpsQjtBQUtFLG9CQUFnQixFQUFFNkIsVUFMcEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsQ0F4QkosRUFvQ0U7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQXBDRixFQXFDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGdFQUFEO0FBQWdCLFFBQUksRUFBRSxxQkFBdEI7QUFBNkMsU0FBSyxFQUFDLG9CQUFuRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxvRUFBRDtBQUNFLFdBQU8sRUFBRW1ILHlCQURYO0FBRUUsa0JBQWMsRUFBRSx3QkFBQ2pFLE1BQUQ7QUFBQSxhQUF5QkEsTUFBTSxDQUFDaEMsS0FBaEM7QUFBQSxLQUZsQjtBQUdFLGtCQUFjLEVBQUUsd0JBQUNnQyxNQUFEO0FBQUEsYUFBeUJBLE1BQU0sQ0FBQzdELEtBQWhDO0FBQUEsS0FIbEI7QUFJRSxpQkFBYSxFQUFFOEUsZUFKakI7QUFLRSxVQUFNLEVBQUUsZ0JBQUM5RSxLQUFELEVBQW9CO0FBQzFCK0UsdUJBQWlCLENBQUMvRSxLQUFELENBQWpCO0FBQ0QsS0FQSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FERixDQXJDRixFQWtERTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBbERGLENBREY7QUFzREQsQ0F6RU07O0dBQU0rSCxZO1VBU0lySCxxRDs7O0tBVEpxSCxZOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDcENiO0FBQ0E7QUFFQTtBQUNBO0FBRUE7QUFTTyxJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxPQUEyQztBQUFBOztBQUFBLE1BQXhDQyxnQkFBd0MsUUFBeENBLGdCQUF3QztBQUNuRSxNQUFNekgsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU0xQyxLQUFpQixHQUFHeUMsTUFBTSxDQUFDekMsS0FBakM7O0FBRm1FLHdCQUduQytDLDhDQUFBLENBQzlCL0MsS0FBSyxDQUFDbUssV0FEd0IsQ0FIbUM7QUFBQTtBQUFBLE1BRzVEQyxRQUg0RDtBQUFBLE1BR2xEQyxXQUhrRDs7QUFPbkV0SCxpREFBQSxDQUFnQixZQUFNO0FBQ3BCLFFBQUcvQyxLQUFLLENBQUNtSyxXQUFOLEtBQXNCQyxRQUF6QixFQUFrQztBQUNoQyxVQUFNekMsTUFBTSxHQUFHOUcsdUZBQXFCLENBQUM7QUFBRXNKLG1CQUFXLEVBQUVDO0FBQWYsT0FBRCxFQUE0QnBLLEtBQTVCLENBQXBDO0FBQ0FZLG9GQUFZLENBQUMrRyxNQUFELENBQVo7QUFDRDtBQUNGLEdBTEQsRUFLRyxDQUFDeUMsUUFBRCxDQUxIO0FBT0EsU0FBT3JILDZDQUFBLENBQWMsWUFBTTtBQUN6QixXQUNFLE1BQUMseURBQUQ7QUFBTSxjQUFRLEVBQUUsa0JBQUN1QyxDQUFEO0FBQUEsZUFBWStFLFdBQVcsQ0FBQy9FLENBQUMsQ0FBQ2dGLE1BQUYsQ0FBU3RJLEtBQVYsQ0FBdkI7QUFBQSxPQUFoQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyxnRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyw4REFBRDtBQUNFLGtCQUFZLEVBQUVoQyxLQUFLLENBQUNtSyxXQUR0QjtBQUVFLGFBQU8sRUFBRUQsZ0JBRlg7QUFHRSxRQUFFLEVBQUMsYUFITDtBQUlFLGlCQUFXLEVBQUMsaUJBSmQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGLENBREYsQ0FERjtBQVlELEdBYk0sRUFhSixDQUFDRSxRQUFELENBYkksQ0FBUDtBQWNELENBNUJNOztHQUFNSCxVO1VBQ0l2SCxxRDs7O0tBREp1SCxVOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDZmI7QUFFQTtBQUNBO0FBRUE7QUFNTyxJQUFNTSxPQUFPLEdBQUcsU0FBVkEsT0FBVSxPQUE2QjtBQUFBOztBQUFBLE1BQTFCdkssS0FBMEIsUUFBMUJBLEtBQTBCOztBQUFBLHdCQUN0QitDLDhDQUFBLENBQWUsS0FBZixDQURzQjtBQUFBO0FBQUEsTUFDM0MwRCxJQUQyQztBQUFBLE1BQ3JDK0QsV0FEcUM7O0FBR2xELFNBQ0UsNERBQ0UsTUFBQywwREFBRDtBQUFjLGVBQVcsRUFBRUEsV0FBM0I7QUFBd0MsUUFBSSxFQUFFL0QsSUFBOUM7QUFBb0QsU0FBSyxFQUFFekcsS0FBM0Q7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBRUU7QUFBSyxXQUFPLEVBQUU7QUFBQSxhQUFNd0ssV0FBVyxDQUFDLENBQUMvRCxJQUFGLENBQWpCO0FBQUEsS0FBZDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywwQ0FBRDtBQUFNLFdBQU8sRUFBRSxVQUFmO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLENBRkYsQ0FERjtBQVVELENBYk07O0dBQU04RCxPOztLQUFBQSxPOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQ1hiO0FBRUE7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFRTyxJQUFNRSxZQUFZLEdBQUcsU0FBZkEsWUFBZSxPQUlIO0FBQUE7O0FBQUEsTUFIdkJ6SyxLQUd1QixRQUh2QkEsS0FHdUI7QUFBQSxNQUZ2QndLLFdBRXVCLFFBRnZCQSxXQUV1QjtBQUFBLE1BRHZCL0QsSUFDdUIsUUFEdkJBLElBQ3VCO0FBQ3ZCLE1BQU15QyxXQUFXLEdBQUduRyxnREFBQSxDQUFpQkUsK0RBQWpCLENBQXBCO0FBQ0EsTUFBTW9HLGNBQWMsR0FBR0MsdUZBQWtCLENBQ3ZDSixXQUR1QyxFQUV2Q2xKLEtBRnVDLEVBR3ZDLE1BSHVDLEVBSXZDLGdCQUp1QyxDQUF6Qzs7QUFGdUIsb0JBU0txRSxvRUFBVSxDQUFDa0YscUVBQWMsQ0FBQ0YsY0FBRCxDQUFmLEVBQWlDLEVBQWpDLEVBQXFDLENBQ3pFckosS0FBSyxDQUFDYyxZQURtRSxFQUV6RWQsS0FBSyxDQUFDMkMsVUFGbUUsQ0FBckMsQ0FUZjtBQUFBLE1BU2Y2QixJQVRlLGVBU2ZBLElBVGU7QUFBQSxNQVNUQyxTQVRTLGVBU1RBLFNBVFM7O0FBY3ZCLE1BQU1LLEdBQUcsR0FBRytFLHlEQUFTLENBQUM7QUFBRTdILFNBQUssRUFBRSxNQUFUO0FBQWlCNkIsU0FBSyxFQUFFO0FBQXhCLEdBQUQsRUFBa0NXLElBQWxDLENBQXJCO0FBQ0EsU0FDRSxNQUFDLDZFQUFEO0FBQ0UsU0FBSyxnQkFBU00sR0FBVCxpQkFEUDtBQUVFLFdBQU8sRUFBRTJCLElBRlg7QUFHRSxZQUFRLEVBQUU7QUFBQSxhQUFNK0QsV0FBVyxDQUFDLEtBQUQsQ0FBakI7QUFBQSxLQUhaO0FBSUUsVUFBTSxFQUFFLENBQ04sTUFBQyw4REFBRDtBQUNFLFdBQUssRUFBRWhDLG1EQUFLLENBQUNDLE1BQU4sQ0FBYUMsU0FBYixDQUF1QkMsSUFEaEM7QUFFRSxnQkFBVSxFQUFDLE9BRmI7QUFHRSxTQUFHLEVBQUMsT0FITjtBQUlFLGFBQU8sRUFBRTtBQUFBLGVBQU02QixXQUFXLENBQUMsS0FBRCxDQUFqQjtBQUFBLE9BSlg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxlQURNLENBSlY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQWVHL0QsSUFBSSxJQUNIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR2lFLG1EQUFRLENBQUMzSixHQUFULENBQWEsVUFBQ3FJLElBQUQ7QUFBQSxXQUNaLE1BQUMsOERBQUQ7QUFBYSxVQUFJLEVBQUVBLElBQW5CO0FBQXlCLFdBQUssRUFBRXBKLEtBQWhDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEWTtBQUFBLEdBQWIsQ0FESCxDQWhCSixDQURGO0FBeUJELENBNUNNOztHQUFNeUssWTtVQWFpQnBHLDREOzs7S0FiakJvRyxZOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDcEJiO0FBQ0E7QUFFQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBYU8sSUFBTUUsV0FBVyxHQUFHLFNBQWRBLFdBQWMsT0FBbUM7QUFBQTs7QUFBQSxNQUFoQzNLLEtBQWdDLFFBQWhDQSxLQUFnQztBQUFBLE1BQXpCb0osSUFBeUIsUUFBekJBLElBQXlCO0FBQzVELE1BQU1GLFdBQVcsR0FBR25HLGdEQUFBLENBQWlCRSwrREFBakIsQ0FBcEI7QUFFQSxNQUFNb0csY0FBYyxHQUFHQyx1RkFBa0IsQ0FDdkNKLFdBRHVDLEVBRXZDbEosS0FGdUMsRUFHdkNvSixJQUFJLENBQUNwSCxLQUhrQyxFQUl2QyxnQkFKdUMsQ0FBekM7O0FBSDRELG9CQVVoQ3FDLG9FQUFVLENBQUNrRixxRUFBYyxDQUFDRixjQUFELENBQWYsRUFBaUMsRUFBakMsRUFBcUMsQ0FDekVySixLQUFLLENBQUNjLFlBRG1FLEVBRXpFZCxLQUFLLENBQUMyQyxVQUZtRSxDQUFyQyxDQVZzQjtBQUFBLE1BVXBENkIsSUFWb0QsZUFVcERBLElBVm9EO0FBQUEsTUFVOUNDLFNBVjhDLGVBVTlDQSxTQVY4Qzs7QUFlNUQsTUFBTW9GLFNBQVMsR0FBRyxTQUFaQSxTQUFZLENBQUNULElBQUQsRUFBcUI7QUFDckMsUUFBTXBILEtBQUssR0FBR3dDLElBQUksR0FBR0EsSUFBSSxDQUFDb0csT0FBUixHQUFrQixJQUFwQzs7QUFFQSxRQUFJeEIsSUFBSSxDQUFDeUIsSUFBTCxLQUFjLE1BQWQsSUFBd0I3SSxLQUE1QixFQUFtQztBQUNqQyxVQUFNOEksT0FBTyxHQUFHLElBQUk3RyxJQUFKLENBQVNwQixRQUFRLENBQUNiLEtBQUQsQ0FBUixHQUFrQixJQUEzQixDQUFoQjtBQUNBLFVBQU0rSSxJQUFJLEdBQUdELE9BQU8sQ0FBQ0UsV0FBUixFQUFiO0FBQ0EsYUFBT0QsSUFBUDtBQUNELEtBSkQsTUFJTztBQUNMLGFBQU8vSSxLQUFLLEdBQUdBLEtBQUgsR0FBVyxnQkFBdkI7QUFDRDtBQUNGLEdBVkQ7O0FBWUEsU0FDRSxNQUFDLDJEQUFEO0FBQVcsV0FBTyxFQUFDLE1BQW5CO0FBQTBCLGtCQUFjLEVBQUMsZUFBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkRBQUQ7QUFBVyxTQUFLLEVBQUUsR0FBbEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUF3Qm9ILElBQUksQ0FBQ3ZGLEtBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsU0FBSyxFQUFFLEdBQWxCO0FBQXVCLFFBQUksRUFBQyxNQUE1QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0dZLFNBQVMsR0FBRyxNQUFDLHlDQUFEO0FBQU0sUUFBSSxFQUFDLE9BQVg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUFILEdBQTJCb0YsU0FBUyxDQUFDVCxJQUFELENBRGhELENBRkYsQ0FERjtBQVFELENBbkNNOztHQUFNdUIsVztVQVVpQnRHLDREOzs7S0FWakJzRyxXOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUNyQmI7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0lBRVFNLE8sR0FBWUMseUMsQ0FBWkQsTzs7QUFNUixJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQUEsMEJBQ2FwSSxnREFBQSxDQUFpQkUsZ0VBQWpCLENBRGI7QUFBQSxNQUNmbUksU0FEZSxxQkFDZkEsU0FEZTtBQUFBLE1BQ0pDLFlBREkscUJBQ0pBLFlBREk7O0FBR3ZCLE1BQU1DLFVBQVUsR0FDZGhJLGdFQUFnQixDQUFDaUksSUFBakIsS0FBMEIsUUFBMUIsR0FBcUNDLDZEQUFyQyxHQUF1REMsOERBRHpEO0FBR0EsTUFBTUMsZ0JBQWdCLEdBQUdwSSxnRUFBZ0IsQ0FBQ2lJLElBQWpCLEtBQTBCLFFBQTFCLEdBQXFDRCxVQUFVLENBQUMsQ0FBRCxDQUFWLENBQWNBLFVBQWQsQ0FBeUIsQ0FBekIsRUFBNEJ6SCxLQUFqRSxHQUF5RXlILFVBQVUsQ0FBQyxDQUFELENBQVYsQ0FBY0EsVUFBZCxDQUF5QixDQUF6QixFQUE0QnpILEtBQTlIO0FBRUFkLGlEQUFBLENBQWdCLFlBQU07QUFDcEJzSSxnQkFBWSxDQUFDSyxnQkFBRCxDQUFaO0FBQ0EsV0FBTztBQUFBLGFBQU1MLFlBQVksQ0FBQ0ssZ0JBQUQsQ0FBbEI7QUFBQSxLQUFQO0FBQ0QsR0FIRCxFQUdHLEVBSEg7QUFLQSxNQUFNakosTUFBTSxHQUFHQyw4REFBUyxFQUF4QjtBQUNBLE1BQU0xQyxLQUFpQixHQUFHeUMsTUFBTSxDQUFDekMsS0FBakM7O0FBZHVCLHdCQWdCb0IrQyw4Q0FBQSxDQUFlLEtBQWYsQ0FoQnBCO0FBQUE7QUFBQSxNQWdCaEI0SSxjQWhCZ0I7QUFBQSxNQWdCQUMsZ0JBaEJBLHdCQWtCdkI7OztBQUNBLFNBQ0UsTUFBQyx5REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixjQUFVLEVBQUMsT0FBM0I7QUFBbUMsU0FBSyxFQUFDLFdBQXpDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQ0UsV0FBTyxFQUFFLG1CQUFNO0FBQ2JBLHNCQUFnQixDQUFDLENBQUNELGNBQUYsQ0FBaEI7QUFDRCxLQUhIO0FBSUUsUUFBSSxFQUFDLE1BSlA7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU1HUCxTQU5ILENBREYsRUFTRSxNQUFDLDZFQUFEO0FBQ0UsU0FBSyxFQUFDLFlBRFI7QUFFRSxXQUFPLEVBQUVPLGNBRlg7QUFHRSxZQUFRLEVBQUU7QUFBQSxhQUFNQyxnQkFBZ0IsQ0FBQyxLQUFELENBQXRCO0FBQUEsS0FIWjtBQUlFLFVBQU0sRUFBRSxDQUNOLE1BQUMsOERBQUQ7QUFDRSxXQUFLLEVBQUVwRCxvREFBSyxDQUFDQyxNQUFOLENBQWFDLFNBQWIsQ0FBdUJDLElBRGhDO0FBRUUsZ0JBQVUsRUFBQyxPQUZiO0FBR0UsU0FBRyxFQUFDLE9BSE47QUFJRSxhQUFPLEVBQUU7QUFBQSxlQUFNaUQsZ0JBQWdCLENBQUMsS0FBRCxDQUF0QjtBQUFBLE9BSlg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxlQURNLENBSlY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQWVFLE1BQUMseUNBQUQ7QUFBTSxvQkFBZ0IsRUFBQyxHQUF2QjtBQUEyQixRQUFJLEVBQUMsTUFBaEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHTixVQUFVLENBQUN2SyxHQUFYLENBQWUsVUFBQ3FLLFNBQUQ7QUFBQSxXQUNkLE1BQUMsT0FBRDtBQUFTLFNBQUcsRUFBRUEsU0FBUyxDQUFDdkgsS0FBeEI7QUFBK0IsU0FBRyxFQUFFdUgsU0FBUyxDQUFDdkgsS0FBOUM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHdUgsU0FBUyxDQUFDRSxVQUFWLENBQXFCdkssR0FBckIsQ0FBeUIsVUFBQzhLLFlBQUQ7QUFBQSxhQUN4QixNQUFDLDJDQUFEO0FBQ0UsV0FBRyxFQUFFQSxZQUFZLENBQUNoSSxLQURwQjtBQUVFLFlBQUksRUFBQyxNQUZQO0FBR0UsZUFBTyxnTUFBRTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ1B3SCw4QkFBWSxDQUFDUSxZQUFZLENBQUNoSSxLQUFkLENBQVo7QUFDQStILGtDQUFnQixDQUFDLENBQUNELGNBQUYsQ0FBaEIsQ0FGTyxDQUdQO0FBQ0E7O0FBSk87QUFBQSx5QkFLREcsbUVBQW1CLENBQUM5TCxLQUFELEVBQVE2TCxZQUFZLENBQUNoSSxLQUFyQixDQUxsQjs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUFGLEVBSFQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQVdHZ0ksWUFBWSxDQUFDaEksS0FYaEIsQ0FEd0I7QUFBQSxLQUF6QixDQURILENBRGM7QUFBQSxHQUFmLENBREgsQ0FmRixDQVRGLENBREYsQ0FERjtBQW1ERCxDQXRFRDs7R0FBTXNILFU7VUFhV3pJLHNEOzs7S0FiWHlJLFU7QUF3RVNBLHlFQUFmOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDN0ZBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFRTyxJQUFNWSxNQUFNLEdBQUcsU0FBVEEsTUFBUyxPQUdIO0FBQUEsTUFGakJDLDZCQUVpQixRQUZqQkEsNkJBRWlCO0FBQUEsTUFEakJoTSxLQUNpQixRQURqQkEsS0FDaUI7QUFDakIsU0FDRSw0REFFSTtBQUNBO0FBQ0E7QUFDQWdNLCtCQUE2QixHQUMzQiw0REFDRSxNQUFDLDJEQUFEO0FBQVMsU0FBSyxFQUFFaE0sS0FBaEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLEVBRUUsTUFBQyxvRkFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRkYsQ0FEMkIsR0FNM0IsNERBQ0UsTUFBQyx1REFBRDtBQUNFLDZCQUF5QixFQUFFQSxLQUFLLENBQUMrRixpQkFEbkM7QUFFRSwrQkFBMkIsRUFBRS9GLEtBQUssQ0FBQ2lHLG1CQUZyQztBQUdFLDhCQUEwQixFQUFFakcsS0FBSyxDQUFDNEMsSUFIcEM7QUFJRSxXQUFPLEVBQUV3RSw4REFKWDtBQUtFLFFBQUksRUFBQyxLQUxQO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQVhOLENBREY7QUF5QkQsQ0E3Qk07S0FBTTJFLE07Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUNiYjtBQUVBO0FBQ0E7QUFDQTtBQU9PLElBQU03TCw2Q0FBNkMsR0FBRyxTQUFoREEsNkNBQWdELENBQzNEeUMsVUFEMkQsRUFFM0Q1QyxjQUYyRCxFQUd4RDtBQUFBOztBQUNILE1BQU1rTSxvQkFBb0IsR0FBR0MsdUdBQW9DLENBQy9Ebk0sY0FEK0QsQ0FBakU7QUFJQSxNQUFNb00scUJBQXFCLEdBQUdqTCxNQUFNLENBQUNDLElBQVAsQ0FBWThLLG9CQUFaLEVBQWtDRyxJQUFsQyxFQUE5Qjs7QUFMRyxtQkFNeUI3RSw0REFBUyxDQUFDNUUsVUFBRCxFQUFhLEVBQWIsQ0FObEM7QUFBQSxNQU1LNkUsZUFOTCxjQU1LQSxlQU5MLEVBT0g7OztBQUNBLE1BQU02RSxXQUFXLEdBQUc3RSxlQUFlLENBQUN6RyxHQUFoQixDQUFvQixVQUFDcUIsTUFBRDtBQUFBLFdBQVlBLE1BQU0sQ0FBQ3VFLE9BQW5CO0FBQUEsR0FBcEIsQ0FBcEI7QUFFQSxNQUFNMkYsYUFBYSxHQUFHSCxxQkFBcUIsQ0FBQyxDQUFELENBQTNDLENBVkcsQ0FXSDtBQUNBOztBQVpHLGtCQWdCQ3BLLHNEQUFRLENBQUN1SyxhQUFELENBaEJUO0FBQUEsTUFjREMsK0JBZEM7QUFBQSxNQWVEL0wsa0NBZkMsaUJBa0JIOzs7QUFsQkcsbUJBc0JDdUIsc0RBQVEsQ0FBQ2tLLG9CQUFvQixDQUFDSyxhQUFELENBQXJCLENBdEJUO0FBQUEsTUFvQkQvTCw0QkFwQkM7QUFBQSxNQXFCREQsK0JBckJDLGtCQXdCSDtBQUNBOzs7QUF6QkcsbUJBMEJ1Q3lCLHNEQUFRLENBQ2hEbUssdUdBQW9DLENBQUNuTSxjQUFELENBRFksQ0ExQi9DO0FBQUEsTUEwQklNLGFBMUJKO0FBQUEsTUEwQm1CRCxnQkExQm5CLGtCQThCSDtBQUNBO0FBQ0E7QUFDQTs7O0FBQ0EsTUFBTW9NLGdCQUFnQixHQUFHQyx5RkFBZSxDQUN0Q0osV0FEc0MsRUFFdENFLCtCQUZzQyxDQUF4QyxDQWxDRyxDQXVDSDtBQUVBO0FBQ0E7O0FBQ0EsTUFBTUcsWUFBWSxHQUFHeEwsTUFBTSxDQUFDeUwsTUFBUCxDQUFjdE0sYUFBZCxDQUFyQjtBQUNBcU0sY0FBWSxDQUFDRSxPQUFiLENBQXFCLEVBQXJCO0FBQ0EsTUFBTWxNLGVBQWUsR0FBR2dNLFlBQVksQ0FBQ0csSUFBYixDQUFrQixHQUFsQixDQUF4QixDQTdDRyxDQThDSDtBQUNBOztBQUNBLE1BQU1wTSwyQ0FBMkMsR0FBRzRMLFdBQVcsQ0FBQ1MsUUFBWixDQUNsRHBNLGVBRGtELENBQXBEO0FBSUEsTUFBTVAsb0NBQW9DLEdBQUdnTSxxQkFBcUIsQ0FBQ3BMLEdBQXRCLENBQzNDLFVBQUNDLElBQUQsRUFBa0I7QUFDaEIsUUFBTUssZ0JBQTBCLEdBQUcwTCxzRkFBbUIsQ0FDcERQLGdCQURvRCxFQUVwRGpNLDRCQUZvRCxFQUdwRFMsSUFIb0QsQ0FBdEQ7QUFNQSxRQUFNSSxtQkFBbUIsR0FBRzRMLGlGQUFjLENBQ3hDM0wsZ0JBRHdDLEVBRXhDZ0wsV0FGd0MsRUFHeENyTCxJQUh3QyxDQUExQztBQU1BLHlHQUNHQSxJQURILEVBQ1U7QUFDTkssc0JBQWdCLEVBQUVBLGdCQURaO0FBRU5ELHlCQUFtQixFQUFFQTtBQUZmLEtBRFY7QUFNRCxHQXBCMEMsQ0FBN0M7QUF1QkEsU0FBTztBQUNMakIsd0NBQW9DLEVBQXBDQSxvQ0FESztBQUVMQyxvQkFBZ0IsRUFBaEJBLGdCQUZLO0FBR0xDLGlCQUFhLEVBQWJBLGFBSEs7QUFJTEMsbUNBQStCLEVBQS9CQSwrQkFKSztBQUtMQyxnQ0FBNEIsRUFBNUJBLDRCQUxLO0FBTUxDLHNDQUFrQyxFQUFsQ0Esa0NBTks7QUFPTEMsK0NBQTJDLEVBQTNDQSwyQ0FQSztBQVFMQyxtQkFBZSxFQUFmQTtBQVJLLEdBQVA7QUFVRCxDQXhGTTs7R0FBTVIsNkM7VUFTaUJxSCxvRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FDcEI5QjtBQUVBO0FBQ0E7QUFDQTtBQUVBO0FBU0E7QUFDQTtBQUNBOztBQUVBLElBQU0wRixLQUFnQyxHQUFHLFNBQW5DQSxLQUFtQyxHQUFNO0FBQUE7O0FBQzdDO0FBQ0EsTUFBTXhLLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7QUFDQSxNQUFNMUMsS0FBaUIsR0FBR3lDLE1BQU0sQ0FBQ3pDLEtBQWpDO0FBRUEsTUFBTWdNLDZCQUE2QixHQUNqQyxDQUFDLENBQUNoTSxLQUFLLENBQUMyQyxVQUFSLElBQXNCLENBQUMsQ0FBQzNDLEtBQUssQ0FBQ2MsWUFEaEM7QUFHQSxTQUNFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0RBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFO0FBQ0UsZUFBVyxFQUFDLFdBRGQ7QUFFRSxRQUFJLEVBQUMsaUJBRlA7QUFHRSxPQUFHLEVBQUMscURBSE47QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsRUFRRSxNQUFDLHFFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHFFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDRDQUFEO0FBQVMsU0FBSyxFQUFDLG1CQUFmO0FBQW1DLGFBQVMsRUFBQyxZQUE3QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxzRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywwRUFBRDtBQUFtQixXQUFPLEVBQUUsaUJBQUN3RSxDQUFEO0FBQUEsYUFBSzRILG1FQUFjLENBQUM1SCxDQUFELENBQW5CO0FBQUEsS0FBNUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsbUVBQUQ7QUFBWSxPQUFHLEVBQUMscURBQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLENBREYsQ0FERixFQVFFLE1BQUMsaUVBQUQ7QUFDRSxpQ0FBNkIsRUFBRTBHLDZCQURqQztBQUVFLFNBQUssRUFBRWhNLEtBRlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQVJGLENBREYsRUFjRSxNQUFDLCtGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFkRixDQVJGLENBREY7QUEyQkQsQ0FuQ0Q7O0dBQU1pTixLO1VBRVd2SyxxRDs7O0tBRlh1SyxLO0FBcUNTQSxvRUFBZjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQ25EQTtBQUFBO0FBQUE7QUFBTyxJQUFNRSxrQkFBa0IsR0FBRyxDQUNoQztBQUNFdEosT0FBSyxFQUFFLFNBRFQ7QUFFRXVKLGFBQVcsRUFBRSxDQUFDLFNBQUQ7QUFGZixDQURnQyxFQUtoQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0V2SixPQUFLLEVBQUUsT0FEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsVUFBRDtBQUZmLENBVGdDLEVBYWhDO0FBQ0V2SixPQUFLLEVBQUUsTUFEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsTUFBRDtBQUZmLENBYmdDLEVBaUJoQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0V2SixPQUFLLEVBQUUsWUFEVDtBQUVFdUosYUFBVyxFQUFFO0FBRmYsQ0FyQmdDLENBQTNCO0FBMkJQLElBQU1DLGdCQUFnQixHQUFHLENBQ3ZCO0FBQ0V4SixPQUFLLEVBQUUsS0FEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsS0FBRDtBQUZmLENBRHVCLEVBS3ZCO0FBQ0V2SixPQUFLLEVBQUUsWUFEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsWUFBRDtBQUZmLENBTHVCLEVBU3ZCO0FBQ0V2SixPQUFLLEVBQUUsU0FEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsU0FBRDtBQUZmLENBVHVCLEVBYXZCO0FBQ0V2SixPQUFLLEVBQUUsUUFEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsUUFBRDtBQUZmLENBYnVCLEVBaUJ2QjtBQUNFdkosT0FBSyxFQUFFLEtBRFQ7QUFFRXVKLGFBQVcsRUFBRSxDQUFDLEtBQUQ7QUFGZixDQWpCdUIsQ0FBekI7QUF1QkEsSUFBTUUsZ0JBQWdCLEdBQUcsQ0FDdkI7QUFDRXpKLE9BQUssRUFBRSxhQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxhQUFEO0FBRmYsQ0FEdUIsRUFLdkI7QUFDRXZKLE9BQUssRUFBRSxPQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxPQUFEO0FBRmYsQ0FMdUIsRUFTdkI7QUFDRXZKLE9BQUssRUFBRSxTQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxTQUFELEVBQVksVUFBWjtBQUZmLENBVHVCLENBQXpCO0FBZUEsSUFBTUcscUJBQXFCLEdBQUcsQ0FDNUI7QUFDRTFKLE9BQUssRUFBRSxNQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxNQUFELEVBQVMsWUFBVCxFQUF1QixZQUF2QixFQUFxQyxpQkFBckM7QUFGZixDQUQ0QixFQUs1QjtBQUNFdkosT0FBSyxFQUFFLGVBRFQ7QUFFRXVKLGFBQVcsRUFBRSxDQUFDLGVBQUQ7QUFGZixDQUw0QixFQVM1QjtBQUNFdkosT0FBSyxFQUFFLE1BRFQ7QUFFRXVKLGFBQVcsRUFBRSxDQUFDLE1BQUQsRUFBUyxPQUFUO0FBRmYsQ0FUNEIsRUFhNUI7QUFDRXZKLE9BQUssRUFBRSxXQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxXQUFEO0FBRmYsQ0FiNEIsRUFpQjVCO0FBQ0V2SixPQUFLLEVBQUUsUUFEVDtBQUVFdUosYUFBVyxFQUFFLENBQUMsUUFBRDtBQUZmLENBakI0QixDQUE5QjtBQXVCQSxJQUFNSSxjQUFjLEdBQUcsQ0FDckI7QUFDRTNKLE9BQUssRUFBRSxLQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxLQUFEO0FBRmYsQ0FEcUIsRUFLckI7QUFDRXZKLE9BQUssRUFBRSxJQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxJQUFEO0FBRmYsQ0FMcUIsRUFTckI7QUFDRXZKLE9BQUssRUFBRSxLQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FBQyxLQUFEO0FBRmYsQ0FUcUIsQ0FBdkI7QUFlQSxJQUFNSyxhQUFhLEdBQUcsQ0FDcEI7QUFDRTVKLE9BQUssRUFBRSxlQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FDWCxxQkFEVyxFQUVYLGNBRlcsRUFHWCw2QkFIVztBQUZmLENBRG9CLEVBU3BCO0FBQ0V2SixPQUFLLEVBQUUsZUFEVDtBQUVFdUosYUFBVyxFQUFFLENBQ1gscUJBRFcsRUFFWCxjQUZXLEVBR1gsNkJBSFc7QUFGZixDQVRvQixFQWlCcEI7QUFDRXZKLE9BQUssRUFBRSxlQURUO0FBRUV1SixhQUFXLEVBQUUsQ0FDWCxxQkFEVyxFQUVYLGNBRlcsRUFHWCw2QkFIVztBQUZmLENBakJvQixFQXlCcEI7QUFDRXZKLE9BQUssRUFBRSxtQkFEVDtBQUVFdUosYUFBVyxFQUFFLENBQ1gseUJBRFcsRUFFWCxjQUZXLEVBR1gsaUNBSFc7QUFGZixDQXpCb0IsQ0FBdEI7QUFtQ08sSUFBTTlCLFVBQVUsR0FBRyxDQUN4QjtBQUNFekgsT0FBSyxFQUFFLFdBRFQ7QUFFRXlILFlBQVUsRUFBRTZCO0FBRmQsQ0FEd0IsRUFLeEI7QUFDRXRKLE9BQUssRUFBRSxTQURUO0FBRUV5SCxZQUFVLEVBQUUrQjtBQUZkLENBTHdCLEVBU3hCO0FBQ0V4SixPQUFLLEVBQUUsU0FEVDtBQUVFeUgsWUFBVSxFQUFFZ0M7QUFGZCxDQVR3QixFQWF4QjtBQUNFekosT0FBSyxFQUFFLGNBRFQ7QUFFRXlILFlBQVUsRUFBRWlDO0FBRmQsQ0Fid0IsRUFpQnhCO0FBQ0UxSixPQUFLLEVBQUUsT0FEVDtBQUVFeUgsWUFBVSxFQUFFa0M7QUFGZCxDQWpCd0IsRUFxQnhCO0FBQ0UzSixPQUFLLEVBQUUsT0FEVDtBQUVFeUgsWUFBVSxFQUFFbUM7QUFGZCxDQXJCd0IsQ0FBbkIiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguOGQyNDViOWE5ZTc0OGJhN2VkNTcuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyBDb2wsIFJvdyB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyBQYXJ0c0Jyb3dzZXIgfSBmcm9tICcuL3BhcnRCcm93c2VyJztcbmltcG9ydCB7IFN0eWxlZFN1Y2Nlc3NJY29uLCBTdHlsZWRFcnJvckljb24gfSBmcm9tICcuLi8uLi9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IHVzZUF2YWlsYmxlQW5kTm90QXZhaWxhYmxlRGF0YXNldFBhcnRzT3B0aW9ucyB9IGZyb20gJy4uLy4uLy4uL2hvb2tzL3VzZUF2YWlsYmxlQW5kTm90QXZhaWxhYmxlRGF0YXNldFBhcnRzT3B0aW9ucyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHtcbiAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLFxuICBjaGFuZ2VSb3V0ZXIsXG59IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscyc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgRGF0YXNldFBhcnRzUHJvcHMge1xuICBwYXJ0XzA6IGFueTtcbiAgcGFydF8xOiBhbnk7XG4gIHBhcnRfMjogYW55O1xufVxuXG5pbnRlcmZhY2UgRGF0YXNldHNCdWlsZGVyUHJvcHMge1xuICBjdXJyZW50RGF0YXNldDogc3RyaW5nO1xuICBxdWVyeTogUXVlcnlQcm9wcztcbiAgY3VycmVudFJ1bk51bWJlcjogc3RyaW5nO1xufVxuXG5leHBvcnQgY29uc3QgRGF0YXNldHNCdWlsZGVyID0gKHtcbiAgY3VycmVudERhdGFzZXQsXG4gIHF1ZXJ5LFxuICBjdXJyZW50UnVuTnVtYmVyLFxufTogRGF0YXNldHNCdWlsZGVyUHJvcHMpID0+IHtcbiAgY29uc3Qge1xuICAgIGF2YWlsYWJsZUFuZE5vdEF2YWlsYWJsZURhdGFzZXRQYXJ0cyxcbiAgICBzZXRTZWxlY3RlZFBhcnRzLFxuICAgIHNlbGVjdGVkUGFydHMsXG4gICAgc2V0TGFzdFNlbGVjdGVkRGF0YXNldFBhcnRWYWx1ZSxcbiAgICBsYXN0U2VsZWN0ZWREYXRhc2V0UGFydFZhbHVlLFxuICAgIHNldExhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb24sXG4gICAgZG9lc0NvbWJpbmF0aW9uT2ZTZWxlY3RlZERhdGFzZXRQYXJ0c0V4aXN0cyxcbiAgICBmdWxsRGF0YXNldE5hbWUsXG4gIH0gPSB1c2VBdmFpbGJsZUFuZE5vdEF2YWlsYWJsZURhdGFzZXRQYXJ0c09wdGlvbnMoXG4gICAgY3VycmVudFJ1bk51bWJlcixcbiAgICBjdXJyZW50RGF0YXNldFxuICApO1xuXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgaWYgKGRvZXNDb21iaW5hdGlvbk9mU2VsZWN0ZWREYXRhc2V0UGFydHNFeGlzdHMpIHtcbiAgICAgIGNoYW5nZVJvdXRlcihcbiAgICAgICAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHsgZGF0YXNldF9uYW1lOiBmdWxsRGF0YXNldE5hbWUgfSwgcXVlcnkpXG4gICAgICApO1xuICAgIH1cbiAgfSwgW2Z1bGxEYXRhc2V0TmFtZV0pO1xuXG4gIHJldHVybiAoXG4gICAgPFJvdz5cbiAgICAgIHthdmFpbGFibGVBbmROb3RBdmFpbGFibGVEYXRhc2V0UGFydHMubWFwKChwYXJ0OiBhbnkpID0+IHtcbiAgICAgICAgY29uc3QgcGFydE5hbWUgPSBPYmplY3Qua2V5cyhwYXJ0KVswXTtcbiAgICAgICAgcmV0dXJuIChcbiAgICAgICAgICA8Q29sPlxuICAgICAgICAgICAgPFBhcnRzQnJvd3NlclxuICAgICAgICAgICAgICByZXN0UGFydHM9e3BhcnRbcGFydE5hbWVdLm5vdEF2YWlsYWJsZUNob2ljZXN9XG4gICAgICAgICAgICAgIHBhcnQ9e3BhcnROYW1lfVxuICAgICAgICAgICAgICByZXN1bHRzTmFtZXM9e3BhcnRbcGFydE5hbWVdLmF2YWlsYWJsZUNob2ljZXN9XG4gICAgICAgICAgICAgIHNldEdyb3VwQnk9e3NldExhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb259XG4gICAgICAgICAgICAgIHNldE5hbWU9e3NldExhc3RTZWxlY3RlZERhdGFzZXRQYXJ0VmFsdWV9XG4gICAgICAgICAgICAgIHNlbGVjdGVkTmFtZT17bGFzdFNlbGVjdGVkRGF0YXNldFBhcnRWYWx1ZX1cbiAgICAgICAgICAgICAgLy9AdHMtaWdub3JlXG4gICAgICAgICAgICAgIG5hbWU9e3NlbGVjdGVkUGFydHNbcGFydE5hbWVdfVxuICAgICAgICAgICAgICBzZXRTZWxlY3RlZFBhcnRzPXtzZXRTZWxlY3RlZFBhcnRzfVxuICAgICAgICAgICAgICBzZWxlY3RlZFBhcnRzPXtzZWxlY3RlZFBhcnRzfVxuICAgICAgICAgICAgLz5cbiAgICAgICAgICA8L0NvbD5cbiAgICAgICAgKTtcbiAgICAgIH0pfVxuICAgICAgPENvbD5cbiAgICAgICAge2RvZXNDb21iaW5hdGlvbk9mU2VsZWN0ZWREYXRhc2V0UGFydHNFeGlzdHMgPyAoXG4gICAgICAgICAgPFN0eWxlZFN1Y2Nlc3NJY29uIC8+XG4gICAgICAgICkgOiAoXG4gICAgICAgICAgPFN0eWxlZEVycm9ySWNvbiAvPlxuICAgICAgICApfVxuICAgICAgPC9Db2w+XG4gICAgPC9Sb3c+XG4gICk7XG59O1xuIiwiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgU2VsZWN0IH0gZnJvbSAnYW50ZCc7XG5cbmltcG9ydCB7IFN0eWxlZFNlbGVjdCB9IGZyb20gJy4uLy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IFN0eWxlZE9wdGlvbkNvbnRlbnQgfSBmcm9tICcuLi8uLi9zdHlsZWRDb21wb25lbnRzJztcbmNvbnN0IHsgT3B0aW9uIH0gPSBTZWxlY3Q7XG5cbmludGVyZmFjZSBQYXJ0c0Jyb3dzZXJQcm9wcyB7XG4gIHNldEdyb3VwQnkoZ3JvdXBCeTogc3RyaW5nKTogdm9pZDtcbiAgc2V0TmFtZShuYW1lOiBzdHJpbmcpOiB2b2lkO1xuICByZXN1bHRzTmFtZXM6IGFueTtcbiAgcmVzdFBhcnRzOiBzdHJpbmdbXTtcbiAgcGFydDogc3RyaW5nO1xuICBuYW1lOiBzdHJpbmcgfCB1bmRlZmluZWQ7XG4gIHNldFNlbGVjdGVkUGFydHMoc2VsZWN0ZWRQYXJ0OiBhbnkpOiB2b2lkO1xuICBzZWxlY3RlZFBhcnRzOiBhbnk7XG4gIHNlbGVjdGVkTmFtZTogYW55O1xufVxuXG5leHBvcnQgY29uc3QgUGFydHNCcm93c2VyID0gKHtcbiAgc2V0TmFtZSxcbiAgc2V0R3JvdXBCeSxcbiAgcmVzdWx0c05hbWVzLFxuICByZXN0UGFydHMsXG4gIHBhcnQsXG4gIG5hbWUsXG4gIHNldFNlbGVjdGVkUGFydHMsXG4gIHNlbGVjdGVkUGFydHMsXG4gIHNlbGVjdGVkTmFtZSxcbn06IFBhcnRzQnJvd3NlclByb3BzKSA9PiB7XG4gIGNvbnN0IFt2YWx1ZSwgc2V0VmFsdWVdID0gdXNlU3RhdGUobmFtZSk7XG4gIGNvbnN0IFtvcGVuU2VsZWN0LCBzZXRTZWxlY3RdID0gdXNlU3RhdGUoZmFsc2UpO1xuXG4gIHJldHVybiAoXG4gICAgPFN0eWxlZFNlbGVjdFxuICAgICAgZHJvcGRvd25NYXRjaFNlbGVjdFdpZHRoPXtmYWxzZX1cbiAgICAgIGRlZmF1bHRWYWx1ZT17bmFtZX1cbiAgICAgIHNlbGVjdGVkPXtzZWxlY3RlZE5hbWUgPT09IHZhbHVlID8gJ3NlbGVjdGVkJyA6ICcnfVxuICAgICAgb25DaGFuZ2U9eyh2YWx1ZTogYW55KSA9PiB7XG4gICAgICAgIHNlbGVjdGVkUGFydHNbcGFydF0gPSB2YWx1ZTtcbiAgICAgICAgc2V0U2VsZWN0ZWRQYXJ0cyhzZWxlY3RlZFBhcnRzKTtcbiAgICAgICAgc2V0R3JvdXBCeShwYXJ0KTtcbiAgICAgICAgc2V0VmFsdWUodmFsdWUpO1xuICAgICAgICBzZXROYW1lKHZhbHVlKTtcbiAgICAgIH19XG4gICAgICBvbkNsaWNrPXsoKSA9PiBzZXRTZWxlY3QoIW9wZW5TZWxlY3QpfVxuICAgICAgb3Blbj17b3BlblNlbGVjdH1cbiAgICA+XG4gICAgICB7cmVzdWx0c05hbWVzLm1hcCgocmVzdWx0OiBzdHJpbmcpID0+IChcbiAgICAgICAgPE9wdGlvbiB2YWx1ZT17cmVzdWx0fSBrZXk9e3Jlc3VsdH0gb25DbGljaz17KCkgPT4gc2V0U2VsZWN0KGZhbHNlKX0+XG4gICAgICAgICAgPFN0eWxlZE9wdGlvbkNvbnRlbnQgYXZhaWxhYmlsaXR5PVwiYXZhaWxhYmxlXCI+XG4gICAgICAgICAgICB7cmVzdWx0fVxuICAgICAgICAgIDwvU3R5bGVkT3B0aW9uQ29udGVudD5cbiAgICAgICAgPC9PcHRpb24+XG4gICAgICApKX1cbiAgICAgIHtyZXN0UGFydHMubWFwKChyZXN1bHQ6IHN0cmluZykgPT4gKFxuICAgICAgICA8T3B0aW9uIGtleT17cmVzdWx0fSB2YWx1ZT17cmVzdWx0fT5cbiAgICAgICAgICA8U3R5bGVkT3B0aW9uQ29udGVudD57cmVzdWx0fTwvU3R5bGVkT3B0aW9uQ29udGVudD5cbiAgICAgICAgPC9PcHRpb24+XG4gICAgICApKX1cbiAgICA8L1N0eWxlZFNlbGVjdD5cbiAgKTtcbn07XG4iLCJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBXcmFwcGVyRGl2IH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRGF0YXNldHNCcm93c2VyIH0gZnJvbSAnLi9kYXRhc2V0c0Jyb3dzaW5nL2RhdGFzZXRzQnJvd3Nlcic7XG5pbXBvcnQgeyBEYXRhc2V0c0J1aWxkZXIgfSBmcm9tICcuL2RhdGFzZXRzQnJvd3NpbmcvZGF0YXNldE5hbWVCdWlsZGVyJztcbmltcG9ydCB7IFJ1bkJyb3dzZXIgfSBmcm9tICcuL3J1bnNCcm93c2VyJztcbmltcG9ydCB7IEx1bWVzZWN0aW9uQnJvd3NlciB9IGZyb20gJy4vbHVtZXNlY3Rpb25Ccm93ZXNlcic7XG5pbXBvcnQgeyBkYXRhU2V0U2VsZWN0aW9ucyB9IGZyb20gJy4uL2NvbnN0YW50cyc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRHJvcGRvd25NZW51IH0gZnJvbSAnLi4vbWVudSc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdXNlQ2hhbmdlUm91dGVyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlQ2hhbmdlUm91dGVyJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7XG4gIGNoYW5nZVJvdXRlcixcbiAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLFxufSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgQnJvd3NlciA9ICgpID0+IHtcbiAgY29uc3QgW2RhdGFzZXRPcHRpb24sIHNldERhdGFzZXRPcHRpb25dID0gdXNlU3RhdGUoXG4gICAgZGF0YVNldFNlbGVjdGlvbnNbMF0udmFsdWVcbiAgKTtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IHJ1bl9udW1iZXIgPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xuICBjb25zdCBkYXRhc2V0X25hbWUgPSBxdWVyeS5kYXRhc2V0X25hbWUgPyBxdWVyeS5kYXRhc2V0X25hbWUgOiAnJztcbiAgY29uc3QgbHVtaSA9IHF1ZXJ5Lmx1bWkgPyBwYXJzZUludChxdWVyeS5sdW1pKSA6IE5hTjtcblxuICBjb25zdCB7IHNldEx1bWlzZWN0aW9uIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcbiAgY29uc3QgW2N1cnJlbnRSdW5OdW1iZXIsIHNldEN1cnJlbnRSdW5OdW1iZXJdID0gdXNlU3RhdGUocnVuX251bWJlcik7XG4gIGNvbnN0IFtjdXJyZW50RGF0YXNldCwgc2V0Q3VycmVudERhdGFzZXRdID0gdXNlU3RhdGU8c3RyaW5nPihkYXRhc2V0X25hbWUpO1xuXG4gIGNvbnN0IGx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIgPSAobHVtaTogbnVtYmVyKSA9PiB7XG4gICAgLy9pbiBtYWluIG5hdmlnYXRpb24gd2hlbiBsdW1pc2VjdGlvbiBpcyBjaGFuZ2VkLCBuZXcgdmFsdWUgaGF2ZSB0byBiZSBzZXQgdG8gdXJsXG4gICAgY2hhbmdlUm91dGVyKGdldENoYW5nZWRRdWVyeVBhcmFtcyh7IGx1bWk6IGx1bWkgfSwgcXVlcnkpKTtcbiAgICAvL3NldEx1bWlzZWN0aW9uIGZyb20gc3RvcmUodXNpbmcgdXNlQ29udGV4dCkgc2V0IGx1bWlzZWN0aW9uIHZhbHVlIGdsb2JhbGx5LlxuICAgIC8vVGhpcyBzZXQgdmFsdWUgaXMgcmVhY2hhYmxlIGZvciBsdW1pc2VjdGlvbiBicm93c2VyIGluIGZyZWUgc2VhcmNoIGRpYWxvZyAoeW91IGNhbiBzZWUgaXQsIHdoZW4gc2VhcmNoIGJ1dHRvbiBuZXh0IHRvIGJyb3dzZXJzIGlzIGNsaWNrZWQpLlxuXG4gICAgLy9Cb3RoIGx1bWlzZWN0aW9uIGJyb3dzZXIgaGF2ZSBkaWZmZXJlbnQgaGFuZGxlcnMsIHRoZXkgaGF2ZSB0byBhY3QgZGlmZmVyZW50bHkgYWNjb3JkaW5nIHRvIHRoZWlyIHBsYWNlOlxuICAgIC8vSU4gVEhFIE1BSU4gTkFWOiBsdW1pc2VjdGlvbiBicm93c2VyIHZhbHVlIGluIHRoZSBtYWluIG5hdmlnYXRpb24gaXMgY2hhbmdlZCwgdGhpcyBIQVZFIHRvIGJlIHNldCB0byB1cmw7XG4gICAgLy9GUkVFIFNFQVJDSCBESUFMT0c6IGx1bWlzZWN0aW9uIGJyb3dzZXIgdmFsdWUgaW4gZnJlZSBzZWFyY2ggZGlhbG9nIGlzIGNoYW5nZWQgaXQgSEFTTidUIHRvIGJlIHNldCB0byB1cmwgaW1tZWRpYXRlbHksIGp1c3Qgd2hlbiBidXR0b24gJ29rJ1xuICAgIC8vaW4gZGlhbG9nIGlzIGNsaWNrZWQgVEhFTiB2YWx1ZSBpcyBzZXQgdG8gdXJsLiBTbywgdXNlQ29udGV4dCBsZXQgdXMgdG8gY2hhbmdlIGx1bWkgdmFsdWUgZ2xvYmFsbHkgd2l0aG91dCBjaGFuZ2luZyB1cmwsIHdoZW4gd2VlIG5vIG5lZWQgdGhhdC5cbiAgICAvL0FuZCBpbiB0aGlzIGhhbmRsZXIgbHVtaSB2YWx1ZSBzZXQgdG8gdXNlQ29udGV4dCBzdG9yZSBpcyB1c2VkIGFzIGluaXRpYWwgbHVtaSB2YWx1ZSBpbiBmcmVlIHNlYXJjaCBkaWFsb2cuXG4gICAgc2V0THVtaXNlY3Rpb24obHVtaSk7XG4gIH07XG5cbiAgaWYgKGN1cnJlbnRSdW5OdW1iZXIgIT09IHF1ZXJ5LnJ1bl9udW1iZXIgfHwgY3VycmVudERhdGFzZXQgIT09IHF1ZXJ5LmRhdGFzZXRfbmFtZSkge1xuICAgIGNoYW5nZVJvdXRlcihcbiAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhcbiAgICAgICAge1xuICAgICAgICAgIHJ1bl9udW1iZXI6IGN1cnJlbnRSdW5OdW1iZXIsXG4gICAgICAgICAgZGF0YXNldF9uYW1lOiBjdXJyZW50RGF0YXNldCxcbiAgICAgICAgfSxcbiAgICAgICAgcXVlcnlcbiAgICAgIClcbiAgICApO1xuICB9XG5cbiAgLy9tYWtlIGNoYW5nZXMgdGhyb3VnaCBjb250ZXh0XG4gIHJldHVybiAoXG4gICAgPEZvcm0+XG4gICAgICA8V3JhcHBlckRpdj5cbiAgICAgICAgPFdyYXBwZXJEaXY+XG4gICAgICAgICAgPFJ1bkJyb3dzZXIgcXVlcnk9e3F1ZXJ5fSBzZXRDdXJyZW50UnVuTnVtYmVyPXtzZXRDdXJyZW50UnVuTnVtYmVyfSAvPlxuICAgICAgICA8L1dyYXBwZXJEaXY+XG4gICAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICAgIHtmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5sdW1pc2VjdGlvbnNfb24gJiYgKFxuICAgICAgICAgICAgPEx1bWVzZWN0aW9uQnJvd3NlclxuICAgICAgICAgICAgICBjdXJyZW50THVtaXNlY3Rpb249e2x1bWl9XG4gICAgICAgICAgICAgIGN1cnJlbnRSdW5OdW1iZXI9e2N1cnJlbnRSdW5OdW1iZXJ9XG4gICAgICAgICAgICAgIGN1cnJlbnREYXRhc2V0PXtjdXJyZW50RGF0YXNldH1cbiAgICAgICAgICAgICAgaGFuZGxlcj17bHVtaXNlY3Rpb25zQ2hhbmdlSGFuZGxlcn1cbiAgICAgICAgICAgICAgY29sb3I9XCJ3aGl0ZVwiXG4gICAgICAgICAgICAvPlxuICAgICAgICAgICl9XG4gICAgICAgIDwvV3JhcHBlckRpdj5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtXG4gICAgICAgICAgbGFiZWxjb2xvcj1cIndoaXRlXCJcbiAgICAgICAgICBsYWJlbD17XG4gICAgICAgICAgICA8RHJvcGRvd25NZW51XG4gICAgICAgICAgICAgIG9wdGlvbnM9e2RhdGFTZXRTZWxlY3Rpb25zfVxuICAgICAgICAgICAgICBhY3Rpb249e3NldERhdGFzZXRPcHRpb259XG4gICAgICAgICAgICAgIGRlZmF1bHRWYWx1ZT17ZGF0YVNldFNlbGVjdGlvbnNbMF19XG4gICAgICAgICAgICAvPlxuICAgICAgICAgIH1cbiAgICAgICAgPlxuICAgICAgICAgIHtkYXRhc2V0T3B0aW9uID09PSBkYXRhU2V0U2VsZWN0aW9uc1swXS52YWx1ZSA/IChcbiAgICAgICAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICAgICAgICA8RGF0YXNldHNCcm93c2VyXG4gICAgICAgICAgICAgICAgc2V0Q3VycmVudERhdGFzZXQ9e3NldEN1cnJlbnREYXRhc2V0fVxuICAgICAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cbiAgICAgICAgICAgICAgLz5cbiAgICAgICAgICAgIDwvV3JhcHBlckRpdj5cbiAgICAgICAgICApIDogKFxuICAgICAgICAgICAgICA8V3JhcHBlckRpdj5cbiAgICAgICAgICAgICAgICA8RGF0YXNldHNCdWlsZGVyXG4gICAgICAgICAgICAgICAgICBjdXJyZW50UnVuTnVtYmVyPXtjdXJyZW50UnVuTnVtYmVyfVxuICAgICAgICAgICAgICAgICAgY3VycmVudERhdGFzZXQ9e2N1cnJlbnREYXRhc2V0fVxuICAgICAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxuICAgICAgICAgICAgICAgIC8+XG4gICAgICAgICAgICAgIDwvV3JhcHBlckRpdj5cbiAgICAgICAgICAgICl9XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICA8L1dyYXBwZXJEaXY+XG4gICAgPC9Gb3JtPlxuICApO1xufTtcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IENvbCwgU2VsZWN0LCBTcGluLCBCdXR0b24sIFJvdyB9IGZyb20gJ2FudGQnO1xuaW1wb3J0IHsgQ2FyZXRSaWdodEZpbGxlZCwgQ2FyZXRMZWZ0RmlsbGVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xuXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XG5pbXBvcnQgeyBnZXRMdW1pc2VjdGlvbnMsIGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7XG4gIFN0eWxlZFNlbGVjdCxcbiAgT3B0aW9uUGFyYWdyYXBoLFxufSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgT3B0aW9uUHJvcHMsIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5cbmNvbnN0IHsgT3B0aW9uIH0gPSBTZWxlY3Q7XG5cbmludGVyZmFjZSBBbGxSdW5zV2l0aEx1bWlQcm9wcyB7XG4gIHJ1bjogc3RyaW5nO1xuICBsdW1pOiBudW1iZXI7XG4gIGRhdGFzZXQ6IHN0cmluZztcbn1cbmludGVyZmFjZSBMdW1lc2VjdGlvbkJyb3dzZXJQcm9wcyB7XG4gIGN1cnJlbnRMdW1pc2VjdGlvbjogbnVtYmVyO1xuICBjdXJyZW50UnVuTnVtYmVyOiBzdHJpbmc7XG4gIGN1cnJlbnREYXRhc2V0OiBzdHJpbmc7XG4gIGhhbmRsZXIobHVtaTogbnVtYmVyKTogdm9pZDtcbiAgY29sb3I/OiBzdHJpbmc7XG59XG5cbmV4cG9ydCBjb25zdCBMdW1lc2VjdGlvbkJyb3dzZXIgPSAoe1xuICBjb2xvcixcbiAgY3VycmVudEx1bWlzZWN0aW9uLFxuICBoYW5kbGVyLFxuICBjdXJyZW50UnVuTnVtYmVyLFxuICBjdXJyZW50RGF0YXNldCxcbn06IEx1bWVzZWN0aW9uQnJvd3NlclByb3BzKSA9PiB7XG4gIC8vMCAtIGl0IHJlcHJlc2VudHMgQUxMIGx1bWlzZWN0aW9ucy4gSWYgbm9uZSBsdW1pc2VjdGlvbiBpcyBzZWxlY3RlZCwgdGhlbiBwbG90cyB3aGljaCBhcmUgZGlzcGxhaWRcbiAgLy9jb25zaXN0IG9mIEFMTCBsdW1pc2VjdGlvbnMuXG4gIGNvbnN0IFtsdW1pc2VjdGlvbnMsIHNldEx1bWlzZWN0aW9uc10gPSBSZWFjdC51c2VTdGF0ZShbXG4gICAgeyBsYWJlbDogJ0FsbCcsIHZhbHVlOiAwIH0sXG4gIF0pO1xuXG4gIGNvbnN0IGN1cnJlbnRfdGltZSA9IG5ldyBEYXRlKCkuZ2V0VGltZSgpO1xuICBjb25zdCBbbm90X29sZGVyX3RoYW4sIHNldF9ub3Rfb2xkZXJfdGhhbl0gPSBSZWFjdC51c2VTdGF0ZShjdXJyZW50X3RpbWUpO1xuXG4gIC8vZ2V0dGluZyBhbGwgcnVuIGx1bWlzZWN0aW9uc1xuICBjb25zdCB7IGRhdGEsIGlzTG9hZGluZywgZXJyb3JzIH0gPSB1c2VSZXF1ZXN0KFxuICAgIGdldEx1bWlzZWN0aW9ucyh7XG4gICAgICBydW5fbnVtYmVyOiBjdXJyZW50UnVuTnVtYmVyLFxuICAgICAgZGF0YXNldF9uYW1lOiBjdXJyZW50RGF0YXNldCxcbiAgICAgIGx1bWk6IC0xLFxuICAgICAgbm90T2xkZXJUaGFuOiBub3Rfb2xkZXJfdGhhbixcbiAgICB9KSxcbiAgICB7fSxcbiAgICBbY3VycmVudFJ1bk51bWJlciwgY3VycmVudERhdGFzZXQsIG5vdF9vbGRlcl90aGFuXVxuICApO1xuICBjb25zdCBhbGxfcnVuc193aXRoX2x1bWkgPSBkYXRhID8gZGF0YS5kYXRhIDogW107XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICAvL2V4dHJhY3RpbmcganVzdCBsdW1pc2VjdGlvbnMgZnJvbSBkYXRhIG9iamVjdFxuICAgIGNvbnN0IGx1bWlzZWN0aW9uc19mcm9tX2FwaTogT3B0aW9uUHJvcHNbXSA9XG4gICAgICBhbGxfcnVuc193aXRoX2x1bWkubGVuZ3RoID4gMFxuICAgICAgICA/IGFsbF9ydW5zX3dpdGhfbHVtaS5tYXAoKHJ1bjogQWxsUnVuc1dpdGhMdW1pUHJvcHMpID0+IHtcbiAgICAgICAgICAgIHJldHVybiB7IGxhYmVsOiBydW4ubHVtaS50b1N0cmluZygpLCB2YWx1ZTogcnVuLmx1bWkgfTtcbiAgICAgICAgICB9KVxuICAgICAgICA6IFtdO1xuICAgIGNvbnN0IGNvcHkgPSBbLi4ubHVtaXNlY3Rpb25zXTtcbiAgICBjb25zdCBhbGxMdW1pcyA9IGNvcHkuY29uY2F0KGx1bWlzZWN0aW9uc19mcm9tX2FwaSk7XG4gICAgc2V0THVtaXNlY3Rpb25zKGFsbEx1bWlzKTtcbiAgfSwgW2FsbF9ydW5zX3dpdGhfbHVtaV0pO1xuXG4gIGNvbnN0IGx1bWlWYWx1ZXMgPSBsdW1pc2VjdGlvbnMubWFwKChsdW1pOiBPcHRpb25Qcm9wcykgPT4gbHVtaS52YWx1ZSk7XG5cbiAgLy8wIGx1bWlzZWN0aW9uIGlzIG5vdCBleGlzdHMsIGl0IGFkZGVkIGFzIHJlcHJlc2VudGF0aW9uIG9mIEFMTCBsdW1pc2VjdGlvbnMuIElmIG5vbmUgb2YgbHVtZXNjdGlvbnMgaXMgc2VsZWN0ZWRcbiAgLy9pdCBtZWFucyB0aGF0IHNob3VsZCBiZSBkaXNwbGFpZCBwbG90cyB3aGljaCBjb25zdGlzdCBvZiBBTEwgbHVtaWVjdGlvbnMuXG4gIC8vVGhlIHNhbWUgc2l0dWF0aW9uIHdoZW4gcnVuIGRvZXNuJ3QgaGF2ZSBsdW1pcyBhdCBhbGwuIEl0IG1lYW5zIHRoYXQgaXQgZGlzcGxheXMgcGxvdHMgb2YgQUxMIEx1bWlzXG4gIGNvbnN0IGN1cnJlbnRMdW1pSW5kZXggPVxuICAgIGx1bWlWYWx1ZXMuaW5kZXhPZihjdXJyZW50THVtaXNlY3Rpb24pID09PSAtMVxuICAgICAgPyAwXG4gICAgICA6IGx1bWlWYWx1ZXMuaW5kZXhPZihjdXJyZW50THVtaXNlY3Rpb24pO1xuICByZXR1cm4gKFxuICAgIDxDb2w+XG4gICAgICA8U3R5bGVkRm9ybUl0ZW0gbGFiZWxjb2xvcj17Y29sb3J9IG5hbWU9eydsdW1pJ30gbGFiZWw9XCJMdW1pXCI+XG4gICAgICAgIDxSb3cganVzdGlmeT1cImNlbnRlclwiIGFsaWduPVwibWlkZGxlXCI+XG4gICAgICAgICAgPENvbD5cbiAgICAgICAgICAgIDxCdXR0b25cbiAgICAgICAgICAgICAgZGlzYWJsZWQ9eyFsdW1pc2VjdGlvbnNbY3VycmVudEx1bWlJbmRleCAtIDFdfVxuICAgICAgICAgICAgICBpY29uPXs8Q2FyZXRMZWZ0RmlsbGVkIC8+fVxuICAgICAgICAgICAgICB0eXBlPVwibGlua1wiXG4gICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcbiAgICAgICAgICAgICAgICBoYW5kbGVyKGx1bWlWYWx1ZXNbY3VycmVudEx1bWlJbmRleCAtIDFdKTtcbiAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgPC9Db2w+XG5cbiAgICAgICAgICA8Q29sPlxuICAgICAgICAgICAgPFN0eWxlZFNlbGVjdFxuICAgICAgICAgICAgICBkcm9wZG93bk1hdGNoU2VsZWN0V2lkdGg9e2ZhbHNlfVxuICAgICAgICAgICAgICB2YWx1ZT17bHVtaVZhbHVlc1tjdXJyZW50THVtaUluZGV4XX1cbiAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBhbnkpID0+IHtcbiAgICAgICAgICAgICAgICBoYW5kbGVyKHBhcnNlSW50KGUpKTtcbiAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgc2hvd1NlYXJjaD17dHJ1ZX1cbiAgICAgICAgICAgID5cbiAgICAgICAgICAgICAge2x1bWlzZWN0aW9ucyAmJlxuICAgICAgICAgICAgICAgIGx1bWlzZWN0aW9ucy5tYXAoKGN1cnJlbnRfbHVtaXNlY3Rpb246IE9wdGlvblByb3BzKSA9PiB7XG4gICAgICAgICAgICAgICAgICByZXR1cm4gKFxuICAgICAgICAgICAgICAgICAgICA8T3B0aW9uXG4gICAgICAgICAgICAgICAgICAgICAgdmFsdWU9e2N1cnJlbnRfbHVtaXNlY3Rpb24udmFsdWV9XG4gICAgICAgICAgICAgICAgICAgICAga2V5PXtjdXJyZW50X2x1bWlzZWN0aW9uLnZhbHVlLnRvU3RyaW5nKCl9XG4gICAgICAgICAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgICAgICAgICB7aXNMb2FkaW5nID8gKFxuICAgICAgICAgICAgICAgICAgICAgICAgPE9wdGlvblBhcmFncmFwaD5cbiAgICAgICAgICAgICAgICAgICAgICAgICAgPFNwaW4gLz5cbiAgICAgICAgICAgICAgICAgICAgICAgIDwvT3B0aW9uUGFyYWdyYXBoPlxuICAgICAgICAgICAgICAgICAgICAgICkgOiAoXG4gICAgICAgICAgICAgICAgICAgICAgICA8cD57Y3VycmVudF9sdW1pc2VjdGlvbi5sYWJlbH08L3A+XG4gICAgICAgICAgICAgICAgICAgICAgKX1cbiAgICAgICAgICAgICAgICAgICAgPC9PcHRpb24+XG4gICAgICAgICAgICAgICAgICApO1xuICAgICAgICAgICAgICAgIH0pfVxuICAgICAgICAgICAgPC9TdHlsZWRTZWxlY3Q+XG4gICAgICAgICAgPC9Db2w+XG4gICAgICAgICAgPENvbD5cbiAgICAgICAgICAgIDxCdXR0b25cbiAgICAgICAgICAgICAgaWNvbj17PENhcmV0UmlnaHRGaWxsZWQgLz59XG4gICAgICAgICAgICAgIGRpc2FibGVkPXshbHVtaXNlY3Rpb25zW2N1cnJlbnRMdW1pSW5kZXggKyAxXX1cbiAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XG4gICAgICAgICAgICAgICAgaGFuZGxlcihsdW1pVmFsdWVzW2N1cnJlbnRMdW1pSW5kZXggKyAxXSk7XG4gICAgICAgICAgICAgIH19XG4gICAgICAgICAgICAvPlxuICAgICAgICAgIDwvQ29sPlxuICAgICAgICA8L1Jvdz5cbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgPC9Db2w+XG4gICk7XG59O1xuIiwiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgTWVudSwgRHJvcGRvd24sIFJvdywgQ29sIH0gZnJvbSAnYW50ZCc7XG5pbXBvcnQgeyBEb3duT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XG5cbmltcG9ydCB7IE9wdGlvblByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XG4gIG9wdGlvbnM6IE9wdGlvblByb3BzW107XG4gIGRlZmF1bHRWYWx1ZTogT3B0aW9uUHJvcHM7XG4gIGFjdGlvbj8odmFsdWU6IGFueSk6IHZvaWQ7XG59XG5cbmV4cG9ydCBjb25zdCBEcm9wZG93bk1lbnUgPSAoeyBvcHRpb25zLCBkZWZhdWx0VmFsdWUsIGFjdGlvbiB9OiBNZW51UHJvcHMpID0+IHtcbiAgY29uc3QgW3ZhbHVlLCBzZXRWYWx1ZV0gPSB1c2VTdGF0ZShkZWZhdWx0VmFsdWUpO1xuICBjb25zdCBwbG90TWVudSA9IChvcHRpb25zOiBPcHRpb25Qcm9wc1tdLCBkZWZhdWx0VmFsdWU6IE9wdGlvblByb3BzKSA9PiAoXG4gICAgPE1lbnU+XG4gICAgICB7b3B0aW9ucy5tYXAoKG9wdGlvbjogT3B0aW9uUHJvcHMpID0+IChcbiAgICAgICAgPE1lbnUuSXRlbVxuICAgICAgICAgIGtleT17b3B0aW9uLnZhbHVlfVxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcbiAgICAgICAgICAgIGFjdGlvbiAmJiBhY3Rpb24ob3B0aW9uLnZhbHVlKTtcbiAgICAgICAgICAgIHNldFZhbHVlKG9wdGlvbik7XG4gICAgICAgICAgfX1cbiAgICAgICAgPlxuICAgICAgICAgIDxkaXY+e29wdGlvbi5sYWJlbH08L2Rpdj5cbiAgICAgICAgPC9NZW51Lkl0ZW0+XG4gICAgICApKX1cbiAgICA8L01lbnU+XG4gICk7XG5cbiAgcmV0dXJuIChcbiAgICA8Um93PlxuICAgICAgPENvbD5cbiAgICAgICAgPERyb3Bkb3duIG92ZXJsYXk9e3Bsb3RNZW51KG9wdGlvbnMsIGRlZmF1bHRWYWx1ZSl9IHRyaWdnZXI9e1snaG92ZXInXX0+XG4gICAgICAgICAgPGEgc3R5bGU9e3sgY29sb3I6ICd3aGl0ZScgfX0+XG4gICAgICAgICAgICB7dmFsdWUubGFiZWx9IDxEb3duT3V0bGluZWQgLz57JyAnfVxuICAgICAgICAgIDwvYT5cbiAgICAgICAgPC9Ecm9wZG93bj5cbiAgICAgIDwvQ29sPlxuICAgIDwvUm93PlxuICApO1xufTtcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5cclxuaW1wb3J0IHsgQ3VzdG9tQ29sLCBDdXN0b21Sb3cgfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgU2VhcmNoTW9kYWwgfSBmcm9tICcuL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IEJyb3dzZXIgfSBmcm9tICcuLi9icm93c2luZyc7XHJcbmltcG9ydCB7IFNlYXJjaEJ1dHRvbiB9IGZyb20gJy4uL3NlYXJjaEJ1dHRvbic7XHJcblxyXG5leHBvcnQgY29uc3QgQXJjaGl2ZU1vZGVIZWFkZXIgPSAoKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcblxyXG4gIGNvbnN0IFtzZWFyY2hfcnVuX251bWJlciwgc2V0U2VhcmNoUnVuTnVtYmVyXSA9IFJlYWN0LnVzZVN0YXRlKHJ1bik7XHJcbiAgY29uc3QgW3NlYXJjaF9kYXRhc2V0X25hbWUsIHNldFNlYXJjaERhdGFzZXROYW1lXSA9IFJlYWN0LnVzZVN0YXRlKFxyXG4gICAgcXVlcnkuZGF0YXNldF9uYW1lXHJcbiAgKTtcclxuICBjb25zdCBbbW9kYWxTdGF0ZSwgc2V0TW9kYWxTdGF0ZV0gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICAvL3doZW4gbW9kYWwgaXMgb3BlbiwgcnVuIG51bWJlciBhbmQgZGF0YXNldCBzZWFyY2ggZmllbGRzIGFyZSBmaWxsZWQgd2l0aCB2YWx1ZXMgZnJvbSBxdWVyeVxyXG4gICAgaWYgKG1vZGFsU3RhdGUpIHtcclxuICAgICAgY29uc3QgcnVuID0gcXVlcnkucnVuX251bWJlciA/IHF1ZXJ5LnJ1bl9udW1iZXIgOiAnJztcclxuICAgICAgc2V0U2VhcmNoRGF0YXNldE5hbWUocXVlcnkuZGF0YXNldF9uYW1lKTtcclxuICAgICAgc2V0U2VhcmNoUnVuTnVtYmVyKHJ1bik7XHJcbiAgICB9XHJcbiAgfSwgW21vZGFsU3RhdGVdKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxDdXN0b21Db2wgZGlzcGxheT1cImZsZXhcIiBhbGlnbml0ZW1zPVwiY2VudGVyXCI+XHJcbiAgICAgIDxTZWFyY2hNb2RhbFxyXG4gICAgICAgIG1vZGFsU3RhdGU9e21vZGFsU3RhdGV9XHJcbiAgICAgICAgc2V0TW9kYWxTdGF0ZT17c2V0TW9kYWxTdGF0ZX1cclxuICAgICAgICBzZXRTZWFyY2hSdW5OdW1iZXI9e3NldFNlYXJjaFJ1bk51bWJlcn1cclxuICAgICAgICBzZXRTZWFyY2hEYXRhc2V0TmFtZT17c2V0U2VhcmNoRGF0YXNldE5hbWV9XHJcbiAgICAgICAgc2VhcmNoX3J1bl9udW1iZXI9e3NlYXJjaF9ydW5fbnVtYmVyfVxyXG4gICAgICAgIHNlYXJjaF9kYXRhc2V0X25hbWU9e3NlYXJjaF9kYXRhc2V0X25hbWV9XHJcbiAgICAgIC8+XHJcbiAgICAgIDxDdXN0b21Sb3cgd2lkdGg9XCJmaXQtY29udGVudFwiPlxyXG4gICAgICAgIDxCcm93c2VyIC8+XHJcbiAgICAgICAgPFNlYXJjaEJ1dHRvbiBvbkNsaWNrPXsoKSA9PiBzZXRNb2RhbFN0YXRlKHRydWUpfSAvPlxyXG4gICAgICA8L0N1c3RvbVJvdz5cclxuICAgIDwvQ3VzdG9tQ29sPlxyXG4gICk7XHJcbn07XHJcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IENvbCB9IGZyb20gJ2FudGQnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgV29ya3NwYWNlcyBmcm9tICcuLi93b3Jrc3BhY2VzJztcbmltcG9ydCB7IEN1c3RvbVJvdyB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgUGxvdFNlYXJjaCB9IGZyb20gJy4uL3Bsb3RzL3Bsb3QvcGxvdFNlYXJjaCc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgV3JhcHBlckRpdiB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IExpdmVNb2RlSGVhZGVyIH0gZnJvbSAnLi9saXZlTW9kZUhlYWRlcic7XG5pbXBvcnQgeyBBcmNoaXZlTW9kZUhlYWRlciB9IGZyb20gJy4vYXJjaGl2ZV9tb2RlX2hlYWRlcic7XG5cbmV4cG9ydCBjb25zdCBDb21wb3NlZFNlYXJjaCA9ICgpID0+IHtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IHNldF9vbl9saXZlX21vZGUgPVxuICAgIHF1ZXJ5LnJ1bl9udW1iZXIgPT09ICcwJyAmJiBxdWVyeS5kYXRhc2V0X25hbWUgPT09ICcvR2xvYmFsL09ubGluZS9BTEwnO1xuXG4gIHJldHVybiAoXG4gICAgPEN1c3RvbVJvd1xuICAgICAgd2lkdGg9XCIxMDAlXCJcbiAgICAgIGRpc3BsYXk9XCJmbGV4XCJcbiAgICAgIGp1c3RpZnljb250ZW50PVwic3BhY2UtYmV0d2VlblwiXG4gICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcbiAgICA+XG4gICAgICB7c2V0X29uX2xpdmVfbW9kZSA/IChcbiAgICAgICAgPExpdmVNb2RlSGVhZGVyIHF1ZXJ5PXtxdWVyeX0gLz5cbiAgICAgICkgOiAoXG4gICAgICAgIDxBcmNoaXZlTW9kZUhlYWRlciAvPlxuICAgICAgKX1cbiAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICA8Q29sPlxuICAgICAgICAgIDxXb3Jrc3BhY2VzIC8+XG4gICAgICAgIDwvQ29sPlxuICAgICAgICA8Q29sPlxuICAgICAgICAgIDxQbG90U2VhcmNoIGlzTG9hZGluZ0ZvbGRlcnM9e2ZhbHNlfSAvPlxuICAgICAgICA8L0NvbD5cbiAgICAgIDwvV3JhcHBlckRpdj5cbiAgICA8L0N1c3RvbVJvdz5cbiAgKTtcbn07XG4iLCJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcbmltcG9ydCBxcyBmcm9tICdxcyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XG5cbmltcG9ydCB7XG4gIFN0eWxlZE1vZGFsLFxuICBSZXN1bHRzV3JhcHBlcixcbn0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cyc7XG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi9ob29rcy91c2VTZWFyY2gnO1xuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi9zdHlsZXMvdGhlbWUnO1xuaW1wb3J0IHsgU2VsZWN0ZWREYXRhIH0gZnJvbSAnLi9zZWxlY3RlZERhdGEnO1xuaW1wb3J0IE5hdiBmcm9tICcuLi9OYXYnO1xuaW1wb3J0IHsgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJztcbmltcG9ydCB7IHJvb3RfdXJsIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5cbmludGVyZmFjZSBGcmVlU2VhY3JoTW9kYWxQcm9wcyB7XG4gIHNldE1vZGFsU3RhdGUoc3RhdGU6IGJvb2xlYW4pOiB2b2lkO1xuICBtb2RhbFN0YXRlOiBib29sZWFuO1xuICBzZWFyY2hfcnVuX251bWJlcjogdW5kZWZpbmVkIHwgc3RyaW5nO1xuICBzZWFyY2hfZGF0YXNldF9uYW1lOiBzdHJpbmcgfCB1bmRlZmluZWQ7XG4gIHNldFNlYXJjaERhdGFzZXROYW1lKGRhdGFzZXRfbmFtZTogYW55KTogdm9pZDtcbiAgc2V0U2VhcmNoUnVuTnVtYmVyKHJ1bl9udW1iZXI6IHN0cmluZyk6IHZvaWQ7XG59XG5cbmNvbnN0IG9wZW5fYV9uZXdfdGFiID0gKHF1ZXJ5OiBzdHJpbmcpID0+IHtcbiAgd2luZG93Lm9wZW4ocXVlcnksICdfYmxhbmsnKTtcbn07XG5cbmV4cG9ydCBjb25zdCBTZWFyY2hNb2RhbCA9ICh7XG4gIHNldE1vZGFsU3RhdGUsXG4gIG1vZGFsU3RhdGUsXG4gIHNlYXJjaF9ydW5fbnVtYmVyLFxuICBzZWFyY2hfZGF0YXNldF9uYW1lLFxuICBzZXRTZWFyY2hEYXRhc2V0TmFtZSxcbiAgc2V0U2VhcmNoUnVuTnVtYmVyLFxufTogRnJlZVNlYWNyaE1vZGFsUHJvcHMpID0+IHtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XG5cbiAgY29uc3QgW2RhdGFzZXROYW1lLCBzZXREYXRhc2V0TmFtZV0gPSB1c2VTdGF0ZShkYXRhc2V0KTtcbiAgY29uc3QgW29wZW5SdW5Jbk5ld1RhYiwgdG9nZ2xlUnVuSW5OZXdUYWJdID0gdXNlU3RhdGUoZmFsc2UpO1xuICBjb25zdCBydW4gPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xuICBjb25zdCBbcnVuTnVtYmVyLCBzZXRSdW5OdW1iZXJdID0gdXNlU3RhdGU8c3RyaW5nPihydW4pO1xuXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgY29uc3QgcnVuID0gcXVlcnkucnVuX251bWJlciA/IHF1ZXJ5LnJ1bl9udW1iZXIgOiAnJztcbiAgICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldCk7XG4gICAgc2V0UnVuTnVtYmVyKHJ1bik7XG4gIH0sIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXJdKTtcblxuICBjb25zdCBvbkNsb3NpbmcgPSAoKSA9PiB7XG4gICAgc2V0TW9kYWxTdGF0ZShmYWxzZSk7XG4gIH07XG5cbiAgY29uc3Qgc2VhcmNoSGFuZGxlciA9IChydW5fbnVtYmVyOiBzdHJpbmcsIGRhdGFzZXRfbmFtZTogc3RyaW5nKSA9PiB7XG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldF9uYW1lKTtcbiAgICBzZXRSdW5OdW1iZXIocnVuX251bWJlcik7XG4gIH07XG5cbiAgY29uc3QgbmF2aWdhdGlvbkhhbmRsZXIgPSAoXG4gICAgc2VhcmNoX2J5X3J1bl9udW1iZXI6IHN0cmluZyxcbiAgICBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmdcbiAgKSA9PiB7XG4gICAgc2V0U2VhcmNoUnVuTnVtYmVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyKTtcbiAgICBzZXRTZWFyY2hEYXRhc2V0TmFtZShzZWFyY2hfYnlfZGF0YXNldF9uYW1lKTtcbiAgfTtcblxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgc2VhcmNoaW5nLCBpc0xvYWRpbmcsIGVycm9ycyB9ID0gdXNlU2VhcmNoKFxuICAgIHNlYXJjaF9ydW5fbnVtYmVyLFxuICAgIHNlYXJjaF9kYXRhc2V0X25hbWVcbiAgKTtcblxuICBjb25zdCBvbk9rID0gYXN5bmMgKCkgPT4ge1xuICAgIGlmIChvcGVuUnVuSW5OZXdUYWIpIHtcbiAgICAgIGNvbnN0IHBhcmFtcyA9IGZvcm0uZ2V0RmllbGRzVmFsdWUoKTtcbiAgICAgIGNvbnN0IG5ld190YWJfcXVlcnlfcGFyYW1zID0gcXMuc3RyaW5naWZ5KFxuICAgICAgICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMocGFyYW1zLCBxdWVyeSlcbiAgICAgICk7XG4gICAgICAvL3Jvb3QgdXJsIGlzIGVuZHMgd2l0aCBmaXJzdCAnPycuIEkgY2FuJ3QgdXNlIGp1c3Qgcm9vdCB1cmwgZnJvbSBjb25maWcuY29uZmlnLCBiZWNhdXNlXG4gICAgICAvL2luIGRldiBlbnYgaXQgdXNlIGxvY2FsaG9zdDo4MDgxL2RxbS9kZXYgKHRoaXMgaXMgb2xkIGJhY2tlbmQgdXJsIGZyb20gd2hlcmUgSSdtIGdldHRpbmcgZGF0YSksXG4gICAgICAvL2J1dCBJIG5lZWQgbG9jYWxob3N0OjMwMDBcbiAgICAgIGNvbnN0IGN1cnJlbnRfcm9vdCA9IHdpbmRvdy5sb2NhdGlvbi5ocmVmLnNwbGl0KCcvPycpWzBdO1xuICAgICAgb3Blbl9hX25ld190YWIoYCR7Y3VycmVudF9yb290fS8/JHtuZXdfdGFiX3F1ZXJ5X3BhcmFtc31gKTtcbiAgICB9IGVsc2Uge1xuICAgICAgYXdhaXQgZm9ybS5zdWJtaXQoKTtcbiAgICB9XG4gICAgb25DbG9zaW5nKCk7XG4gIH07XG5cbiAgY29uc3QgW2Zvcm1dID0gRm9ybS51c2VGb3JtKCk7XG5cbiAgcmV0dXJuIChcbiAgICA8U3R5bGVkTW9kYWxcbiAgICAgIHRpdGxlPVwiU2VhcmNoIGRhdGFcIlxuICAgICAgdmlzaWJsZT17bW9kYWxTdGF0ZX1cbiAgICAgIG9uQ2FuY2VsPXsoKSA9PiBvbkNsb3NpbmcoKX1cbiAgICAgIGZvb3Rlcj17W1xuICAgICAgICA8U3R5bGVkQnV0dG9uXG4gICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn1cbiAgICAgICAgICBiYWNrZ3JvdW5kPVwid2hpdGVcIlxuICAgICAgICAgIGtleT1cIkNsb3NlXCJcbiAgICAgICAgICBvbkNsaWNrPXsoKSA9PiBvbkNsb3NpbmcoKX1cbiAgICAgICAgPlxuICAgICAgICAgIENsb3NlXG4gICAgICAgIDwvU3R5bGVkQnV0dG9uPixcbiAgICAgICAgPFN0eWxlZEJ1dHRvbiBrZXk9XCJPS1wiIG9uQ2xpY2s9e29uT2t9PlxuICAgICAgICAgIE9LXG4gICAgICAgIDwvU3R5bGVkQnV0dG9uPixcbiAgICAgIF19XG4gICAgPlxuICAgICAge21vZGFsU3RhdGUgJiYgKFxuICAgICAgICA8PlxuICAgICAgICAgIDxOYXZcbiAgICAgICAgICAgIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXI9e3NlYXJjaF9ydW5fbnVtYmVyfVxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtzZWFyY2hfZGF0YXNldF9uYW1lfVxuICAgICAgICAgICAgZGVmYXVsdERhdGFzZXROYW1lPXtkYXRhc2V0TmFtZX1cbiAgICAgICAgICAgIGRlZmF1bHRSdW5OdW1iZXI9e3J1bk51bWJlcn1cbiAgICAgICAgICAgIGhhbmRsZXI9e25hdmlnYXRpb25IYW5kbGVyfVxuICAgICAgICAgICAgdHlwZT1cInRvcFwiXG4gICAgICAgICAgLz5cbiAgICAgICAgICA8U2VsZWN0ZWREYXRhXG4gICAgICAgICAgICBmb3JtPXtmb3JtfVxuICAgICAgICAgICAgZGF0YXNldF9uYW1lPXtkYXRhc2V0TmFtZX1cbiAgICAgICAgICAgIHJ1bl9udW1iZXI9e3J1bk51bWJlcn1cbiAgICAgICAgICAgIHRvZ2dsZVJ1bkluTmV3VGFiPXt0b2dnbGVSdW5Jbk5ld1RhYn1cbiAgICAgICAgICAgIG9wZW5SdW5Jbk5ld1RhYj17b3BlblJ1bkluTmV3VGFifVxuICAgICAgICAgIC8+XG4gICAgICAgICAge3NlYXJjaGluZyA/IChcbiAgICAgICAgICAgIDxSZXN1bHRzV3JhcHBlcj5cbiAgICAgICAgICAgICAgPFNlYXJjaFJlc3VsdHNcbiAgICAgICAgICAgICAgICBoYW5kbGVyPXtzZWFyY2hIYW5kbGVyfVxuICAgICAgICAgICAgICAgIGlzTG9hZGluZz17aXNMb2FkaW5nfVxuICAgICAgICAgICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxuICAgICAgICAgICAgICAgIGVycm9ycz17ZXJyb3JzfVxuICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgPC9SZXN1bHRzV3JhcHBlcj5cbiAgICAgICAgICApIDogKFxuICAgICAgICAgICAgPFJlc3VsdHNXcmFwcGVyIC8+XG4gICAgICAgICAgKX1cbiAgICAgICAgPC8+XG4gICAgICApfVxuICAgIDwvU3R5bGVkTW9kYWw+XG4gICk7XG59O1xuIiwiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgQnV0dG9uLCBUb29sdGlwLCBTcGluLCBUeXBvZ3JhcGh5IH0gZnJvbSAnYW50ZCc7XG5pbXBvcnQgeyBQYXVzZU91dGxpbmVkLCBQbGF5Q2lyY2xlT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XG5cbmltcG9ydCB7XG4gIEN1c3RvbUNvbCxcbiAgQ3VzdG9tRGl2LFxuICBDdXN0b21Gb3JtLFxuICBDdXRvbUZvcm1JdGVtLFxufSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vc3R5bGVzL3RoZW1lJztcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XG5pbXBvcnQgeyBGb3JtYXRQYXJhbXNGb3JBUEkgfSBmcm9tICcuLi9wbG90cy9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuaW1wb3J0IHsgUXVlcnlQcm9wcywgSW5mb1Byb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgbWFpbl9ydW5faW5mbyB9IGZyb20gJy4uL2NvbnN0YW50cyc7XG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgZ2V0X2xhYmVsIH0gZnJvbSAnLi4vdXRpbHMnO1xuY29uc3QgeyBUaXRsZSB9ID0gVHlwb2dyYXBoeTtcblxuaW50ZXJmYWNlIExpdmVNb2RlSGVhZGVyUHJvcHMge1xuICBxdWVyeTogUXVlcnlQcm9wcztcbn1cblxuZXhwb3J0IGNvbnN0IExpdmVNb2RlSGVhZGVyID0gKHsgcXVlcnkgfTogTGl2ZU1vZGVIZWFkZXJQcm9wcykgPT4ge1xuICBjb25zdCB7IHVwZGF0ZSwgc2V0X3VwZGF0ZSB9ID0gdXNlVXBkYXRlTGl2ZU1vZGUoKTtcbiAgY29uc3QgZ2xvYmFsU3RhdGUgPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcbiAgcmV0dXJuIChcbiAgICA8PlxuICAgICAgPEN1c3RvbUZvcm0gZGlzcGxheT1cImZsZXhcIj5cbiAgICAgICAge21haW5fcnVuX2luZm8ubWFwKChpbmZvOiBJbmZvUHJvcHMpID0+IHtcbiAgICAgICAgICBjb25zdCBwYXJhbXNfZm9yX2FwaSA9IEZvcm1hdFBhcmFtc0ZvckFQSShcbiAgICAgICAgICAgIGdsb2JhbFN0YXRlLFxuICAgICAgICAgICAgcXVlcnksXG4gICAgICAgICAgICBpbmZvLnZhbHVlLFxuICAgICAgICAgICAgJy9ITFQvRXZlbnRJbmZvJ1xuICAgICAgICAgICk7XG5cbiAgICAgICAgICBjb25zdCB7IGRhdGEsIGlzTG9hZGluZyB9ID0gdXNlUmVxdWVzdChcbiAgICAgICAgICAgIGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSxcbiAgICAgICAgICAgIHt9LFxuICAgICAgICAgICAgW3F1ZXJ5LmRhdGFzZXRfbmFtZSwgcXVlcnkucnVuX251bWJlcl1cbiAgICAgICAgICApO1xuICAgICAgICAgIHJldHVybiAoXG4gICAgICAgICAgICA8Q3V0b21Gb3JtSXRlbVxuICAgICAgICAgICAgICBzcGFjZT1cIjhcIlxuICAgICAgICAgICAgICB3aWR0aD1cImZpdC1jb250ZW50XCJcbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5jb21tb24ud2hpdGV9XG4gICAgICAgICAgICAgIG5hbWU9e2luZm8ubGFiZWx9XG4gICAgICAgICAgICAgIGxhYmVsPXtpbmZvLmxhYmVsfVxuICAgICAgICAgICAgPlxuICAgICAgICAgICAgICA8VGl0bGVcbiAgICAgICAgICAgICAgICBsZXZlbD17NH1cbiAgICAgICAgICAgICAgICBzdHlsZT17e1xuICAgICAgICAgICAgICAgICAgY29sb3I6IGAke1xuICAgICAgICAgICAgICAgICAgICB1cGRhdGVcbiAgICAgICAgICAgICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xuICAgICAgICAgICAgICAgICAgICAgIDogdGhlbWUuY29sb3JzLm5vdGlmaWNhdGlvbi5lcnJvclxuICAgICAgICAgICAgICAgICAgfWAsXG4gICAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgPlxuICAgICAgICAgICAgICAgIHtpc0xvYWRpbmcgPyA8U3BpbiBzaXplPVwic21hbGxcIiAvPiA6IGdldF9sYWJlbChpbmZvLCBkYXRhKX1cbiAgICAgICAgICAgICAgPC9UaXRsZT5cbiAgICAgICAgICAgIDwvQ3V0b21Gb3JtSXRlbT5cbiAgICAgICAgICApO1xuICAgICAgICB9KX1cbiAgICAgIDwvQ3VzdG9tRm9ybT5cbiAgICAgIDxDdXN0b21Db2xcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJmbGV4LWVuZFwiXG4gICAgICAgIGRpc3BsYXk9XCJmbGV4XCJcbiAgICAgICAgYWxpZ25pdGVtcz1cImNlbnRlclwiXG4gICAgICAgIHRleHR0cmFuc2Zvcm09XCJ1cHBlcmNhc2VcIlxuICAgICAgICBjb2xvcj17XG4gICAgICAgICAgdXBkYXRlXG4gICAgICAgICAgICA/IHRoZW1lLmNvbG9ycy5ub3RpZmljYXRpb24uc3VjY2Vzc1xuICAgICAgICAgICAgOiB0aGVtZS5jb2xvcnMubm90aWZpY2F0aW9uLmVycm9yXG4gICAgICAgIH1cbiAgICAgID5cbiAgICAgICAgTGl2ZSBNb2RlXG4gICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+XG4gICAgICAgICAgPFRvb2x0aXAgdGl0bGU9e2BVcGRhdGluZyBtb2RlIGlzICR7dXBkYXRlID8gJ29uJyA6ICdvZmYnfWB9PlxuICAgICAgICAgICAgPEJ1dHRvblxuICAgICAgICAgICAgICB0eXBlPVwicHJpbWFyeVwiXG4gICAgICAgICAgICAgIHNoYXBlPVwiY2lyY2xlXCJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xuICAgICAgICAgICAgICAgIHNldF91cGRhdGUoIXVwZGF0ZSk7XG4gICAgICAgICAgICAgIH19XG4gICAgICAgICAgICAgIGljb249e3VwZGF0ZSA/IDxQYXVzZU91dGxpbmVkIC8+IDogPFBsYXlDaXJjbGVPdXRsaW5lZCAvPn1cbiAgICAgICAgICAgID48L0J1dHRvbj5cbiAgICAgICAgICA8L1Rvb2x0aXA+XG4gICAgICAgIDwvQ3VzdG9tRGl2PlxuICAgICAgPC9DdXN0b21Db2w+XG4gICAgPC8+XG4gICk7XG59O1xuIiwiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgUm93LCBDb2wgfSBmcm9tICdhbnRkJztcblxuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgTHVtZXNlY3Rpb25Ccm93c2VyIH0gZnJvbSAnLi4vYnJvd3NpbmcvbHVtZXNlY3Rpb25Ccm93ZXNlcic7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0sIFNlbGVjdGVkRGF0YUNvbCB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuaW1wb3J0IHtcbiAgY2hhbmdlUm91dGVyLFxuICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMsXG59IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgUmFkaW9CdXR0b25zR3JvdXAgfSBmcm9tICcuLi9yYWRpb0J1dHRvbnNHcm91cCc7XG5pbXBvcnQgeyBPcHRpb25Qcm9wcyB9IGZyb20gJ2FudGQvbGliL21lbnRpb25zJztcblxuaW50ZXJmYWNlIFNlbGVjdGVkRGF0YVByb3BzIHtcbiAgZGF0YXNldF9uYW1lOiBzdHJpbmc7XG4gIHJ1bl9udW1iZXI6IHN0cmluZztcbiAgZm9ybTogYW55O1xuICB0b2dnbGVSdW5Jbk5ld1RhYihvcGVuOiBib29sZWFuKTogdm9pZDtcbiAgb3BlblJ1bkluTmV3VGFiOiBib29sZWFuO1xufVxuXG5jb25zdCBvcGVuX2luX2FfbmV3X3RhYl9vcHRpb25zID0gW1xuICB7XG4gICAgdmFsdWU6IHRydWUsXG4gICAgbGFiZWw6ICdZZXMnLFxuICB9LFxuICB7XG4gICAgdmFsdWU6IGZhbHNlLFxuICAgIGxhYmVsOiAnTm8nLFxuICB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IFNlbGVjdGVkRGF0YSA9ICh7XG4gIC8vcm91dGVyIG9rXG4gIGRhdGFzZXRfbmFtZSxcbiAgcnVuX251bWJlcixcbiAgZm9ybSxcbiAgdG9nZ2xlUnVuSW5OZXdUYWIsXG4gIG9wZW5SdW5Jbk5ld1RhYixcbn06IFNlbGVjdGVkRGF0YVByb3BzKSA9PiB7XG4gIGNvbnN0IHsgbHVtaXNlY3Rpb24sIHNldEx1bWlzZWN0aW9uIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IGx1bWlzZWN0aW9uc0NoYW5nZUhhbmRsZXIgPSAobHVtaTogbnVtYmVyKSA9PiB7XG4gICAgLy93ZSBzZXQgbHVtaXNlY3Rpb24gaW4gaW5zZUNvbnRleHQgc3RvcmUgaW4gb3JkZXIgdG8gc2F2ZSBhIGl0J3MgdmFsdWUuXG4gICAgLy9XaGVuIGZvcm0gaXMgc3VibWl0dGVkKG9uRmluaXNoLi4uKShjbGlja2VkIGJ1dHRvbiBcIk9LXCIgaW4gZGlhbG9nKSwgdGhlblxuICAgIC8vdXJsIGlzIGNoYW5nZWRcbiAgICBzZXRMdW1pc2VjdGlvbihsdW1pKTtcbiAgfTtcblxuICByZXR1cm4gKFxuICAgIDxGb3JtXG4gICAgICBmb3JtPXtmb3JtfVxuICAgICAgb25GaW5pc2g9eyhwYXJhbXMpID0+IHtcbiAgICAgICAgLy93aGVuIE9LIGlzIGNsaWNrZWQsIHJ1biBudW1iZXIsIGRhdGFzZXQgYW5kIGx1bWkgcGFyYW1zIGluIHVybCBpcyBjaGFuZ2VkLlxuICAgICAgICBjaGFuZ2VSb3V0ZXIoZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHBhcmFtcywgcXVlcnkpKTtcbiAgICAgIH19XG4gICAgICBmaWVsZHM9e1tcbiAgICAgICAgeyBuYW1lOiAnZGF0YXNldF9uYW1lJywgdmFsdWU6IGRhdGFzZXRfbmFtZSB9LFxuICAgICAgICB7IG5hbWU6ICdydW5fbnVtYmVyJywgdmFsdWU6IHJ1bl9udW1iZXIgfSxcbiAgICAgICAgeyBuYW1lOiAnbHVtaScsIHZhbHVlOiBsdW1pc2VjdGlvbiB9LFxuICAgICAgXX1cbiAgICA+XG4gICAgICA8aHIgLz5cbiAgICAgIDxSb3c+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPXsnZGF0YXNldF9uYW1lJ30gbGFiZWw9XCJEYXRhc2V0IG5hbWVcIj5cbiAgICAgICAgICA8U2VsZWN0ZWREYXRhQ29sPntkYXRhc2V0X25hbWV9PC9TZWxlY3RlZERhdGFDb2w+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICA8L1Jvdz5cbiAgICAgIDxSb3c+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPXsncnVuX251bWJlcid9IGxhYmVsPVwiUnVuIG51bWJlclwiPlxuICAgICAgICAgIDxTZWxlY3RlZERhdGFDb2w+e3J1bl9udW1iZXJ9PC9TZWxlY3RlZERhdGFDb2w+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICA8L1Jvdz5cbiAgICAgIHtmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5sdW1pc2VjdGlvbnNfb24gJiYgKFxuICAgICAgICA8Um93PlxuICAgICAgICAgIDxDb2w+XG4gICAgICAgICAgICA8THVtZXNlY3Rpb25Ccm93c2VyXG4gICAgICAgICAgICAgIGNvbG9yPVwiYmxhY2tcIlxuICAgICAgICAgICAgICBoYW5kbGVyPXtsdW1pc2VjdGlvbnNDaGFuZ2VIYW5kbGVyfVxuICAgICAgICAgICAgICBjdXJyZW50THVtaXNlY3Rpb249e2x1bWlzZWN0aW9ufVxuICAgICAgICAgICAgICBjdXJyZW50RGF0YXNldD17ZGF0YXNldF9uYW1lfVxuICAgICAgICAgICAgICBjdXJyZW50UnVuTnVtYmVyPXtydW5fbnVtYmVyfVxuICAgICAgICAgICAgLz5cbiAgICAgICAgICA8L0NvbD5cbiAgICAgICAgPC9Sb3c+XG4gICAgICApfVxuICAgICAgPGhyIC8+XG4gICAgICA8Um93PlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT17J29wZW5faW5fYV9uZXdfYV90YWInfSBsYWJlbD1cIk9wZW4gaW4gYSBuZXcgdGFiP1wiPlxuICAgICAgICAgIDxSYWRpb0J1dHRvbnNHcm91cFxuICAgICAgICAgICAgb3B0aW9ucz17b3Blbl9pbl9hX25ld190YWJfb3B0aW9uc31cbiAgICAgICAgICAgIGdldE9wdGlvbkxhYmVsPXsob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4gb3B0aW9uLmxhYmVsfVxuICAgICAgICAgICAgZ2V0T3B0aW9uVmFsdWU9eyhvcHRpb246IE9wdGlvblByb3BzKSA9PiBvcHRpb24udmFsdWV9XG4gICAgICAgICAgICBjdXJyZW50X3ZhbHVlPXtvcGVuUnVuSW5OZXdUYWJ9XG4gICAgICAgICAgICBhY3Rpb249eyh2YWx1ZTogYm9vbGVhbikgPT4ge1xuICAgICAgICAgICAgICB0b2dnbGVSdW5Jbk5ld1RhYih2YWx1ZSk7XG4gICAgICAgICAgICB9fVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICA8L1Jvdz5cbiAgICAgIDxociAvPlxuICAgIDwvRm9ybT5cbiAgKTtcbn07XG4iLCJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkU2VhcmNoIH0gZnJvbSAnLi4vLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHtcbiAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLFxuICBjaGFuZ2VSb3V0ZXIsXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscyc7XG5cbmludGVyZmFjZSBQbG90U2VhcmNoUHJvcHMge1xuICBpc0xvYWRpbmdGb2xkZXJzOiBib29sZWFuO1xufVxuXG5leHBvcnQgY29uc3QgUGxvdFNlYXJjaCA9ICh7IGlzTG9hZGluZ0ZvbGRlcnMgfTogUGxvdFNlYXJjaFByb3BzKSA9PiB7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcbiAgY29uc3QgW3Bsb3ROYW1lLCBzZXRQbG90TmFtZV0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmcgfCB1bmRlZmluZWQ+KFxuICAgIHF1ZXJ5LnBsb3Rfc2VhcmNoXG4gICk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZihxdWVyeS5wbG90X3NlYXJjaCAhPT0gcGxvdE5hbWUpe1xuICAgICAgY29uc3QgcGFyYW1zID0gZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHsgcGxvdF9zZWFyY2g6IHBsb3ROYW1lIH0sIHF1ZXJ5KTtcbiAgICAgIGNoYW5nZVJvdXRlcihwYXJhbXMpO1xuICAgIH1cbiAgfSwgW3Bsb3ROYW1lXSk7XG5cbiAgcmV0dXJuIFJlYWN0LnVzZU1lbW8oKCkgPT4ge1xuICAgIHJldHVybiAoXG4gICAgICA8Rm9ybSBvbkNoYW5nZT17KGU6IGFueSkgPT4gc2V0UGxvdE5hbWUoZS50YXJnZXQudmFsdWUpfT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtPlxuICAgICAgICAgIDxTdHlsZWRTZWFyY2hcbiAgICAgICAgICAgIGRlZmF1bHRWYWx1ZT17cXVlcnkucGxvdF9zZWFyY2h9XG4gICAgICAgICAgICBsb2FkaW5nPXtpc0xvYWRpbmdGb2xkZXJzfVxuICAgICAgICAgICAgaWQ9XCJwbG90X3NlYXJjaFwiXG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIHBsb3QgbmFtZVwiXG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgIDwvRm9ybT5cbiAgICApO1xuICB9LCBbcGxvdE5hbWVdKTtcbn07XG4iLCJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5cbmltcG9ydCB7IEluZm8gfSBmcm9tICcuLi9pbmZvJztcbmltcG9ydCB7IFJ1bkluZm9Nb2RhbCB9IGZyb20gJy4vcnVuSW5mb01vZGFsJztcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5pbXBvcnQgeyBSdW5JbmZvSWNvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuXG5pbnRlcmZhY2UgUnVuSW5mb1Byb3BzIHtcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XG59XG5cbmV4cG9ydCBjb25zdCBSdW5JbmZvID0gKHsgcXVlcnkgfTogUnVuSW5mb1Byb3BzKSA9PiB7XG4gIGNvbnN0IFtvcGVuLCB0b2dnbGVNb2RhbF0gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XG5cbiAgcmV0dXJuIChcbiAgICA8PlxuICAgICAgPFJ1bkluZm9Nb2RhbCB0b2dnbGVNb2RhbD17dG9nZ2xlTW9kYWx9IG9wZW49e29wZW59IHF1ZXJ5PXtxdWVyeX0gLz5cbiAgICAgIDxkaXYgb25DbGljaz17KCkgPT4gdG9nZ2xlTW9kYWwoIW9wZW4pfT5cbiAgICAgICAgPEluZm8gY29udGVudD17J1J1biBpbmZvJ30+XG4gICAgICAgICAgPFJ1bkluZm9JY29uIC8+XG4gICAgICAgIDwvSW5mbz5cbiAgICAgIDwvZGl2PlxuICAgIDwvPlxuICApO1xufTtcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcblxuaW1wb3J0IHsgU3R5bGVkTW9kYWwgfSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBTdHlsZWRCdXR0b24gfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IHRoZW1lIH0gZnJvbSAnLi4vLi4vc3R5bGVzL3RoZW1lJztcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5pbXBvcnQgeyBSdW5JbmZvSXRlbSB9IGZyb20gJy4vcnVuU3RhcnRUaW1lU3RhbXAnO1xuaW1wb3J0IHsgcnVuX2luZm8gfSBmcm9tICcuLi9jb25zdGFudHMnO1xuaW1wb3J0IHsgRm9ybWF0UGFyYW1zRm9yQVBJIH0gZnJvbSAnLi4vcGxvdHMvcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi9ob29rcy91c2VSZXF1ZXN0JztcbmltcG9ydCB7IGdldF9qcm9vdF9wbG90IH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBnZXRfbGFiZWwgfSBmcm9tICcuLi91dGlscyc7XG5cbmludGVyZmFjZSBSdW5JbmZvTW9kYWxQcm9wcyB7XG4gIHRvZ2dsZU1vZGFsKHZhbHVlOiBib29sZWFuKTogdm9pZDtcbiAgcXVlcnk6IFF1ZXJ5UHJvcHM7XG4gIG9wZW46IGJvb2xlYW47XG59XG5cbmV4cG9ydCBjb25zdCBSdW5JbmZvTW9kYWwgPSAoe1xuICBxdWVyeSxcbiAgdG9nZ2xlTW9kYWwsXG4gIG9wZW4sXG59OiBSdW5JbmZvTW9kYWxQcm9wcykgPT4ge1xuICBjb25zdCBnbG9iYWxTdGF0ZSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xuICBjb25zdCBwYXJhbXNfZm9yX2FwaSA9IEZvcm1hdFBhcmFtc0ZvckFQSShcbiAgICBnbG9iYWxTdGF0ZSxcbiAgICBxdWVyeSxcbiAgICAnaVJ1bicsXG4gICAgJy9ITFQvRXZlbnRJbmZvJ1xuICApO1xuXG4gIGNvbnN0IHsgZGF0YSwgaXNMb2FkaW5nIH0gPSB1c2VSZXF1ZXN0KGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSwge30sIFtcbiAgICBxdWVyeS5kYXRhc2V0X25hbWUsXG4gICAgcXVlcnkucnVuX251bWJlcixcbiAgXSk7XG4gIFxuICBjb25zdCBydW4gPSBnZXRfbGFiZWwoeyB2YWx1ZTogJ2lSdW4nLCBsYWJlbDogJ1J1bicgfSwgZGF0YSk7XG4gIHJldHVybiAoXG4gICAgPFN0eWxlZE1vZGFsXG4gICAgICB0aXRsZT17YFJ1biAke3J1bn0gaW5mb3JtYXRpb25gfVxuICAgICAgdmlzaWJsZT17b3Blbn1cbiAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVNb2RhbChmYWxzZSl9XG4gICAgICBmb290ZXI9e1tcbiAgICAgICAgPFN0eWxlZEJ1dHRvblxuICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XG4gICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcbiAgICAgICAgICBrZXk9XCJDbG9zZVwiXG4gICAgICAgICAgb25DbGljaz17KCkgPT4gdG9nZ2xlTW9kYWwoZmFsc2UpfVxuICAgICAgICA+XG4gICAgICAgICAgQ2xvc2VcbiAgICAgICAgPC9TdHlsZWRCdXR0b24+LFxuICAgICAgXX1cbiAgICA+XG4gICAgICB7b3BlbiAmJiAoXG4gICAgICAgIDxkaXY+XG4gICAgICAgICAge3J1bl9pbmZvLm1hcCgoaW5mbykgPT4gKFxuICAgICAgICAgICAgPFJ1bkluZm9JdGVtIGluZm89e2luZm99IHF1ZXJ5PXtxdWVyeX0gLz5cbiAgICAgICAgICApKX1cbiAgICAgICAgPC9kaXY+XG4gICAgICApfVxuICAgIDwvU3R5bGVkTW9kYWw+XG4gICk7XG59O1xuIiwiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgU3BpbiB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgZ2V0X2pyb290X3Bsb3QgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcbmltcG9ydCB7IEZvcm1hdFBhcmFtc0ZvckFQSSB9IGZyb20gJy4uL3Bsb3RzL3Bsb3Qvc2luZ2xlUGxvdC91dGlscyc7XG5pbXBvcnQgeyBDdXN0b21Db2wsIEN1c3RvbVJvdyB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuXG5pbnRlcmZhY2UgSW5mb1Byb3BzIHtcbiAgdmFsdWU6IHN0cmluZztcbiAgbGFiZWw6IHN0cmluZztcbiAgdHlwZT86IHN0cmluZztcbn1cblxuaW50ZXJmYWNlIFJ1bkluZm9Qcm9wcyB7XG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xuICBpbmZvOiBJbmZvUHJvcHM7XG59XG5cbmV4cG9ydCBjb25zdCBSdW5JbmZvSXRlbSA9ICh7IHF1ZXJ5LCBpbmZvIH06IFJ1bkluZm9Qcm9wcykgPT4ge1xuICBjb25zdCBnbG9iYWxTdGF0ZSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xuXG4gIGNvbnN0IHBhcmFtc19mb3JfYXBpID0gRm9ybWF0UGFyYW1zRm9yQVBJKFxuICAgIGdsb2JhbFN0YXRlLFxuICAgIHF1ZXJ5LFxuICAgIGluZm8udmFsdWUsXG4gICAgJy9ITFQvRXZlbnRJbmZvJ1xuICApO1xuXG4gIGNvbnN0IHsgZGF0YSwgaXNMb2FkaW5nIH0gPSB1c2VSZXF1ZXN0KGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSwge30sIFtcbiAgICBxdWVyeS5kYXRhc2V0X25hbWUsXG4gICAgcXVlcnkucnVuX251bWJlcixcbiAgXSk7XG5cbiAgY29uc3QgZ2V0X2xhYmVsID0gKGluZm86IEluZm9Qcm9wcykgPT4ge1xuICAgIGNvbnN0IHZhbHVlID0gZGF0YSA/IGRhdGEuZlN0cmluZyA6IG51bGw7XG5cbiAgICBpZiAoaW5mby50eXBlID09PSAndGltZScgJiYgdmFsdWUpIHtcbiAgICAgIGNvbnN0IG1pbGlzZWMgPSBuZXcgRGF0ZShwYXJzZUludCh2YWx1ZSkgKiAxMDAwKTtcbiAgICAgIGNvbnN0IHRpbWUgPSBtaWxpc2VjLnRvVVRDU3RyaW5nKCk7XG4gICAgICByZXR1cm4gdGltZTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHZhbHVlID8gdmFsdWUgOiAnTm8gaW5mb3JtYXRpb24nO1xuICAgIH1cbiAgfTtcblxuICByZXR1cm4gKFxuICAgIDxDdXN0b21Sb3cgZGlzcGxheT1cImZsZXhcIiBqdXN0aWZ5Y29udGVudD1cInNwYWNlLWJldHdlZW5cIj5cbiAgICAgIDxDdXN0b21Db2wgc3BhY2U9eycxJ30+e2luZm8ubGFiZWx9PC9DdXN0b21Db2w+XG4gICAgICA8Q3VzdG9tQ29sIHNwYWNlPXsnMSd9IGJvbGQ9XCJ0cnVlXCI+XG4gICAgICAgIHtpc0xvYWRpbmcgPyA8U3BpbiBzaXplPVwic21hbGxcIiAvPiA6IGdldF9sYWJlbChpbmZvKX1cbiAgICAgIDwvQ3VzdG9tQ29sPlxuICAgIDwvQ3VzdG9tUm93PlxuICApO1xufTtcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyB3b3Jrc3BhY2VzIGFzIG9mZmxpbmVXb3Jza3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29mZmxpbmUnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XG5pbXBvcnQgeyBTdHlsZWRNb2RhbCB9IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCBGb3JtIGZyb20gJ2FudGQvbGliL2Zvcm0vRm9ybSc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBzZXRXb3Jrc3BhY2VUb1F1ZXJ5IH0gZnJvbSAnLi91dGlscyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi9zdHlsZXMvdGhlbWUnO1xuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuXG5jb25zdCB7IFRhYlBhbmUgfSA9IFRhYnM7XG5cbmludGVyZmFjZSBXb3JzcGFjZVByb3BzIHtcbiAgbGFiZWw6IHN0cmluZztcbiAgd29ya3NwYWNlczogYW55O1xufVxuY29uc3QgV29ya3NwYWNlcyA9ICgpID0+IHtcbiAgY29uc3QgeyB3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcblxuICBjb25zdCB3b3Jrc3BhY2VzID1cbiAgICBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gb25saW5lV29ya3NwYWNlIDogb2ZmbGluZVdvcnNrcGFjZTtcbiAgICBcbiAgY29uc3QgaW5pdGlhbFdvcmtzcGFjZSA9IGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScgPyB3b3Jrc3BhY2VzWzBdLndvcmtzcGFjZXNbMV0ubGFiZWwgOiB3b3Jrc3BhY2VzWzBdLndvcmtzcGFjZXNbM10ubGFiZWxcblxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xuICAgIHNldFdvcmtzcGFjZShpbml0aWFsV29ya3NwYWNlKVxuICAgIHJldHVybiAoKSA9PiBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcbiAgfSwgW10pXG5cbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IFtvcGVuV29ya3NwYWNlcywgdG9nZ2xlV29ya3NwYWNlc10gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XG5cbiAgLy8gbWFrZSBhIHdvcmtzcGFjZSBzZXQgZnJvbSBjb250ZXh0XG4gIHJldHVybiAoXG4gICAgPEZvcm0+XG4gICAgICA8U3R5bGVkRm9ybUl0ZW0gbGFiZWxjb2xvcj1cIndoaXRlXCIgbGFiZWw9XCJXb3Jrc3BhY2VcIj5cbiAgICAgICAgPEJ1dHRvblxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcbiAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcbiAgICAgICAgICB9fVxuICAgICAgICAgIHR5cGU9XCJsaW5rXCJcbiAgICAgICAgPlxuICAgICAgICAgIHt3b3Jrc3BhY2V9XG4gICAgICAgIDwvQnV0dG9uPlxuICAgICAgICA8U3R5bGVkTW9kYWxcbiAgICAgICAgICB0aXRsZT1cIldvcmtzcGFjZXNcIlxuICAgICAgICAgIHZpc2libGU9e29wZW5Xb3Jrc3BhY2VzfVxuICAgICAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICBmb290ZXI9e1tcbiAgICAgICAgICAgIDxTdHlsZWRCdXR0b25cbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn1cbiAgICAgICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcbiAgICAgICAgICAgICAga2V5PVwiQ2xvc2VcIlxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgQ2xvc2VcbiAgICAgICAgICAgIDwvU3R5bGVkQnV0dG9uPixcbiAgICAgICAgICBdfVxuICAgICAgICA+XG4gICAgICAgICAgPFRhYnMgZGVmYXVsdEFjdGl2ZUtleT1cIjFcIiB0eXBlPVwiY2FyZFwiPlxuICAgICAgICAgICAge3dvcmtzcGFjZXMubWFwKCh3b3Jrc3BhY2U6IFdvcnNwYWNlUHJvcHMpID0+IChcbiAgICAgICAgICAgICAgPFRhYlBhbmUga2V5PXt3b3Jrc3BhY2UubGFiZWx9IHRhYj17d29ya3NwYWNlLmxhYmVsfT5cbiAgICAgICAgICAgICAgICB7d29ya3NwYWNlLndvcmtzcGFjZXMubWFwKChzdWJXb3Jrc3BhY2U6IGFueSkgPT4gKFxuICAgICAgICAgICAgICAgICAgPEJ1dHRvblxuICAgICAgICAgICAgICAgICAgICBrZXk9e3N1YldvcmtzcGFjZS5sYWJlbH1cbiAgICAgICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxuICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXthc3luYyAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgc2V0V29ya3NwYWNlKHN1YldvcmtzcGFjZS5sYWJlbCk7XG4gICAgICAgICAgICAgICAgICAgICAgdG9nZ2xlV29ya3NwYWNlcyghb3BlbldvcmtzcGFjZXMpO1xuICAgICAgICAgICAgICAgICAgICAgIC8vaWYgd29ya3NwYWNlIGlzIHNlbGVjdGVkLCBmb2xkZXJfcGF0aCBpbiBxdWVyeSBpcyBzZXQgdG8gJycuIFRoZW4gd2UgY2FuIHJlZ29uaXplXG4gICAgICAgICAgICAgICAgICAgICAgLy90aGF0IHdvcmtzcGFjZSBpcyBzZWxlY3RlZCwgYW5kIHdlZSBuZWVkIHRvIGZpbHRlciB0aGUgZm9yc3QgbGF5ZXIgb2YgZm9sZGVycy5cbiAgICAgICAgICAgICAgICAgICAgICBhd2FpdCBzZXRXb3Jrc3BhY2VUb1F1ZXJ5KHF1ZXJ5LCBzdWJXb3Jrc3BhY2UubGFiZWwpO1xuICAgICAgICAgICAgICAgICAgICB9fVxuICAgICAgICAgICAgICAgICAgPlxuICAgICAgICAgICAgICAgICAgICB7c3ViV29ya3NwYWNlLmxhYmVsfVxuICAgICAgICAgICAgICAgICAgPC9CdXR0b24+XG4gICAgICAgICAgICAgICAgKSl9XG4gICAgICAgICAgICAgIDwvVGFiUGFuZT5cbiAgICAgICAgICAgICkpfVxuICAgICAgICAgIDwvVGFicz5cbiAgICAgICAgPC9TdHlsZWRNb2RhbD5cbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgPC9Gb3JtPlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgV29ya3NwYWNlcztcbiIsImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcblxuaW1wb3J0IHsgbmF2aWdhdGlvbkhhbmRsZXIgfSBmcm9tICcuLi8uLi91dGlscy9wYWdlcyc7XG5pbXBvcnQgeyBSdW5JbmZvIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy9ydW5JbmZvJztcbmltcG9ydCB7IENvbXBvc2VkU2VhcmNoIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2NvbXBvc2VkU2VhcmNoJztcbmltcG9ydCBOYXYgZnJvbSAnLi4vLi4vY29tcG9uZW50cy9OYXYnO1xuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4vaW50ZXJmYWNlcyc7XG5cbmludGVyZmFjZSBIZWFkZXJQcm9wcyB7XG4gIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkOiBib29sZWFuO1xuICBxdWVyeTogUXVlcnlQcm9wcztcbn1cblxuZXhwb3J0IGNvbnN0IEhlYWRlciA9ICh7XG4gIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkLFxuICBxdWVyeSxcbn06IEhlYWRlclByb3BzKSA9PiB7XG4gIHJldHVybiAoXG4gICAgPD5cbiAgICAgIHtcbiAgICAgICAgLy9pZiBhbGwgZnVsbCBzZXQgaXMgc2VsZWN0ZWQ6IGRhdGFzZXQgbmFtZSBhbmQgcnVuIG51bWJlciwgdGhlbiByZWd1bGFyIHNlYXJjaCBmaWVsZCBpcyBub3QgdmlzaWJsZS5cbiAgICAgICAgLy9JbnN0ZWFkLCBydW4gYW5kIGRhdGFzZXQgYnJvd3NlciBpcyBpcyBkaXNwbGF5ZWQuXG4gICAgICAgIC8vUmVndWxhciBzZWFyY2ggZmllbGRzIGFyZSBkaXNwbGF5ZWQganVzdCBpbiB0aGUgbWFpbiBwYWdlLlxuICAgICAgICBpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCA/IChcbiAgICAgICAgICA8PlxuICAgICAgICAgICAgPFJ1bkluZm8gcXVlcnk9e3F1ZXJ5fSAvPlxuICAgICAgICAgICAgPENvbXBvc2VkU2VhcmNoIC8+XG4gICAgICAgICAgPC8+XG4gICAgICAgICkgOiAoXG4gICAgICAgICAgPD5cbiAgICAgICAgICAgIDxOYXZcbiAgICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj17cXVlcnkuc2VhcmNoX3J1bl9udW1iZXJ9XG4gICAgICAgICAgICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT17cXVlcnkuc2VhcmNoX2RhdGFzZXRfbmFtZX1cbiAgICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb249e3F1ZXJ5Lmx1bWl9XG4gICAgICAgICAgICAgIGhhbmRsZXI9e25hdmlnYXRpb25IYW5kbGVyfVxuICAgICAgICAgICAgICB0eXBlPVwidG9wXCJcbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgPC8+XG4gICAgICAgIClcbiAgICAgIH1cbiAgICA8Lz5cbiAgKTtcbn07XG4iLCJpbXBvcnQgeyB1c2VTdGF0ZSB9IGZyb20gJ3JlYWN0JztcblxuaW1wb3J0IHsgdXNlU2VhcmNoIH0gZnJvbSAnLi91c2VTZWFyY2gnO1xuaW1wb3J0IHsgZ2V0RGF0YXNldFBhcnRzIH0gZnJvbSAnLi4vY29tcG9uZW50cy92aWV3RGV0YWlsc01lbnUvdXRpbHMnO1xuaW1wb3J0IHtcbiAgZ2V0QXZhaWxhYmxlQ2hvaWNlcyxcbiAgZ2V0UmVzdE9wdGlvbnMsXG4gIGdldERhdGFzZXROYW1lU3BsaXRCeVNsYXNoSW50b09iamVjdCxcbn0gZnJvbSAnLi4vY29tcG9uZW50cy9icm93c2luZy91dGlscyc7XG5pbXBvcnQgeyBEYXRhc2V0UGFydHNQcm9wcyB9IGZyb20gJy4uL2NvbXBvbmVudHMvYnJvd3NpbmcvZGF0YXNldHNCcm93c2luZy9kYXRhc2V0TmFtZUJ1aWxkZXInO1xuXG5leHBvcnQgY29uc3QgdXNlQXZhaWxibGVBbmROb3RBdmFpbGFibGVEYXRhc2V0UGFydHNPcHRpb25zID0gKFxuICBydW5fbnVtYmVyOiBzdHJpbmcsXG4gIGN1cnJlbnREYXRhc2V0OiBhbnlcbikgPT4ge1xuICBjb25zdCBzZWxlY3RlZERhdGFzZXRQYXJ0cyA9IGdldERhdGFzZXROYW1lU3BsaXRCeVNsYXNoSW50b09iamVjdChcbiAgICBjdXJyZW50RGF0YXNldFxuICApO1xuXG4gIGNvbnN0IGRhdGFzZXRQYXJ0c1Bvc2l0aW9ucyA9IE9iamVjdC5rZXlzKHNlbGVjdGVkRGF0YXNldFBhcnRzKS5zb3J0KCk7XG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkIH0gPSB1c2VTZWFyY2gocnVuX251bWJlciwgJycpO1xuICAvL2FsbERhdGFzZXRzIGFyZSBhbGwgcG9zc2libGUgZGF0YXNldHNcbiAgY29uc3QgYWxsRGF0YXNldHMgPSByZXN1bHRzX2dyb3VwZWQubWFwKChyZXN1bHQpID0+IHJlc3VsdC5kYXRhc2V0KTtcblxuICBjb25zdCBmaXJzdFBvc2l0aW9uID0gZGF0YXNldFBhcnRzUG9zaXRpb25zWzBdO1xuICAvL2xhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb246IGlzIFBPU0lUSU9OIG9mIGxhc3Qgc2VsZWN0ZWQgZGF0YXNldCBwYXJ0XG4gIC8vbGFzdFNlbGVjdGVkRGF0YXNldFBhcnRQb3NpdGlvbiBpcyB1c2UgZm9yIGdyb3VwaW5nIGFsbCBkYXRhc2V0IHBhcnRzIHBvc3NpYmxlIHZhcmlhbnRzLlxuICBjb25zdCBbXG4gICAgbGFzdFNlbGVjdGVkRGF0YXNldFBhcnRQb3NpdGlvbixcbiAgICBzZXRMYXN0U2VsZWN0ZWREYXRhc2V0UGFydFBvc2l0aW9uLFxuICBdID0gdXNlU3RhdGUoZmlyc3RQb3NpdGlvbik7XG5cbiAgLy9sYXN0U2VsZWN0ZWREYXRhc2V0UGFydE9wdGlvbjogaXMgVkFMVUUgb2YgbGFzdCBzZWxlY3RlZCBkYXRhc2V0IHBhcnRcbiAgY29uc3QgW1xuICAgIGxhc3RTZWxlY3RlZERhdGFzZXRQYXJ0VmFsdWUsXG4gICAgc2V0TGFzdFNlbGVjdGVkRGF0YXNldFBhcnRWYWx1ZSxcbiAgXSA9IHVzZVN0YXRlKHNlbGVjdGVkRGF0YXNldFBhcnRzW2ZpcnN0UG9zaXRpb25dKTtcblxuICAvL3NlbGVjdGVkUGFydHM6IGlzIFNMRUNURUQgZGF0YXNldCBwYXJ0cywgZnJvbSB3aG9tIGNvdWxkIGJlIGZvcm1lZCBmdWxsIGRhdGFzZXRuYW1lXG4gIC8vIGJ5IGRlZmF1dCBzZWxlY3RlZFBhcnRzIGlzIGZvcm1lZCBmcm9tIGN1cnJlbnREYXRhc2V0XG4gIGNvbnN0IFtzZWxlY3RlZFBhcnRzLCBzZXRTZWxlY3RlZFBhcnRzXSA9IHVzZVN0YXRlPERhdGFzZXRQYXJ0c1Byb3BzPihcbiAgICBnZXREYXRhc2V0TmFtZVNwbGl0QnlTbGFzaEludG9PYmplY3QoY3VycmVudERhdGFzZXQpXG4gICk7XG5cbiAgLy9hbGxEYXRhc2V0cyBpcyBzdHJpbmcgYXJyYXkuIE9uZSBzdHJpbmcgZnJvbSB0aGlzIGFycmF5IGlzIEZVTEwgZGF0YXNldCBuYW1lLiBXZSBuZWVkIHRvXG4gIC8vc2VwYXJhdGVkIGVhY2ggZGF0YXNldCBuYW1lIHRvIHBhcnRzLiBPbmUgcGFydCBvZiBkYXRhc2V0IG5hbWUgaW4gRlVMTCBzdHJpbmcgaXMgc2VwYXJhdGVkIGJ5IHNsYXNoLlxuICAvL2dldERhdGFzZXRQYXJ0cyBmdW5jdGlvbiBzZXBhcmF0ZXMgZGF0YXNldCBuYW1lcyB0byBwYXJ0cyBhbmQgZ3JvdXAgdGhlbSBieSBMQVNUIFNFTEVDVEVEIERBVEFTRVQgUEFSVCBQT1NJVElPTi5cbiAgLy9nZXREYXRhc2V0UGFydHMgcmV0dXJucyBMQVNUIFNFTEVDVEVEIFBPU0lUSU9OIFZBTFVFIGFuZCBpdCBwb3NzaWJsZSBjb21iaW5hdGlvbnMgd2l0aCBvdGhlciBwYXJ0c1xuICBjb25zdCBwYXJ0c09iamVjdEFycmF5ID0gZ2V0RGF0YXNldFBhcnRzKFxuICAgIGFsbERhdGFzZXRzLFxuICAgIGxhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb25cbiAgKTtcblxuICAvL2Zyb20gYWxsIHNlbGVjdGVkIGRhdGFzZXQgbmFtZSdzIHBhcnRzIHdlIGZvcm0gZnVsbCBkYXRhc2V0IG5hbWUuXG5cbiAgLy9WYWx1ZXMgb2Ygc2VsZWN0ZWQgZGF0YXNldCBwYXJ0cyBhcmUgaW4gZGF0YXNldFBhcnRzIGFycmF5XG4gIC8vVGhlIGZpcnN0IGVsZW1lbnQgb2YgYXJyYXkgaXMgZW1wdHkgc3RyaW5nLCBiZWNhdXNlIGRhdGFzZXQgbmFtZSBzaG91bGQgc3RhcnQgd2l0aCBzbGFzaC5cbiAgY29uc3QgZGF0YXNldFBhcnRzID0gT2JqZWN0LnZhbHVlcyhzZWxlY3RlZFBhcnRzKTtcbiAgZGF0YXNldFBhcnRzLnVuc2hpZnQoJycpO1xuICBjb25zdCBmdWxsRGF0YXNldE5hbWUgPSBkYXRhc2V0UGFydHMuam9pbignLycpO1xuICAvL1dlIGNoZWNrIGlzIGRhdGFzZXQgbmFtZSBjb21iaW5lZCBmcm9tIHBhcnRzIGlzIGV4aXRzIGluIGFsbCBwb3NzaWJsZSBkYXRhc2V0IG5hbWVzLlxuICAvLyByZW5hbWUgZG9lc0NvbWJpbmF0aW9uT2ZTZWxlY3RlZERhdGFzZXRQYXJ0c0V4aXN0cyB0byBkYXRhc2V0RXhpc3RzIG9yIHJlc3VsdGluZ0RhdGFzZXROYW1lQ29tYmluYXRpb25FeGlzdHNcbiAgY29uc3QgZG9lc0NvbWJpbmF0aW9uT2ZTZWxlY3RlZERhdGFzZXRQYXJ0c0V4aXN0cyA9IGFsbERhdGFzZXRzLmluY2x1ZGVzKFxuICAgIGZ1bGxEYXRhc2V0TmFtZVxuICApO1xuXG4gIGNvbnN0IGF2YWlsYWJsZUFuZE5vdEF2YWlsYWJsZURhdGFzZXRQYXJ0cyA9IGRhdGFzZXRQYXJ0c1Bvc2l0aW9ucy5tYXAoXG4gICAgKHBhcnQ6IHN0cmluZykgPT4ge1xuICAgICAgY29uc3QgYXZhaWxhYmxlQ2hvaWNlczogc3RyaW5nW10gPSBnZXRBdmFpbGFibGVDaG9pY2VzKFxuICAgICAgICBwYXJ0c09iamVjdEFycmF5LFxuICAgICAgICBsYXN0U2VsZWN0ZWREYXRhc2V0UGFydFZhbHVlLFxuICAgICAgICBwYXJ0XG4gICAgICApO1xuXG4gICAgICBjb25zdCBub3RBdmFpbGFibGVDaG9pY2VzID0gZ2V0UmVzdE9wdGlvbnMoXG4gICAgICAgIGF2YWlsYWJsZUNob2ljZXMsXG4gICAgICAgIGFsbERhdGFzZXRzLFxuICAgICAgICBwYXJ0XG4gICAgICApO1xuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBbcGFydF06IHtcbiAgICAgICAgICBhdmFpbGFibGVDaG9pY2VzOiBhdmFpbGFibGVDaG9pY2VzLFxuICAgICAgICAgIG5vdEF2YWlsYWJsZUNob2ljZXM6IG5vdEF2YWlsYWJsZUNob2ljZXMsXG4gICAgICAgIH0sXG4gICAgICB9O1xuICAgIH1cbiAgKTtcblxuICByZXR1cm4ge1xuICAgIGF2YWlsYWJsZUFuZE5vdEF2YWlsYWJsZURhdGFzZXRQYXJ0cyxcbiAgICBzZXRTZWxlY3RlZFBhcnRzLFxuICAgIHNlbGVjdGVkUGFydHMsXG4gICAgc2V0TGFzdFNlbGVjdGVkRGF0YXNldFBhcnRWYWx1ZSxcbiAgICBsYXN0U2VsZWN0ZWREYXRhc2V0UGFydFZhbHVlLFxuICAgIHNldExhc3RTZWxlY3RlZERhdGFzZXRQYXJ0UG9zaXRpb24sXG4gICAgZG9lc0NvbWJpbmF0aW9uT2ZTZWxlY3RlZERhdGFzZXRQYXJ0c0V4aXN0cyxcbiAgICBmdWxsRGF0YXNldE5hbWUsXG4gIH07XG59O1xuIiwiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IE5leHRQYWdlIH0gZnJvbSAnbmV4dCc7XG5pbXBvcnQgSGVhZCBmcm9tICduZXh0L2hlYWQnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgVG9vbHRpcCB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQge1xuICBTdHlsZWRIZWFkZXIsXG4gIFN0eWxlZExheW91dCxcbiAgU3R5bGVkRGl2LFxuICBTdHlsZWRMb2dvV3JhcHBlcixcbiAgU3R5bGVkTG9nbyxcbiAgU3R5bGVkTG9nb0Rpdixcbn0gZnJvbSAnLi4vc3R5bGVzL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRm9sZGVyUGF0aFF1ZXJ5LCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgYmFja1RvTWFpblBhZ2UgfSBmcm9tICcuLi91dGlscy9wYWdlcyc7XG5pbXBvcnQgeyBIZWFkZXIgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaGVhZGVyJztcbmltcG9ydCB7IENvbnRlbnRTd2l0Y2hpbmcgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9jb25zdGVudF9zd2l0Y2hpbmcnO1xuXG5jb25zdCBJbmRleDogTmV4dFBhZ2U8Rm9sZGVyUGF0aFF1ZXJ5PiA9ICgpID0+IHtcbiAgLy8gV2UgZ3JhYiB0aGUgcXVlcnkgZnJvbSB0aGUgVVJMOlxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XG5cbiAgY29uc3QgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQgPVxuICAgICEhcXVlcnkucnVuX251bWJlciAmJiAhIXF1ZXJ5LmRhdGFzZXRfbmFtZTtcblxuICByZXR1cm4gKFxuICAgIDxTdHlsZWREaXY+XG4gICAgICA8SGVhZD5cbiAgICAgICAgPHNjcmlwdFxuICAgICAgICAgIGNyb3NzT3JpZ2luPVwiYW5vbnltb3VzXCJcbiAgICAgICAgICB0eXBlPVwidGV4dC9qYXZhc2NyaXB0XCJcbiAgICAgICAgICBzcmM9XCIuL2pzcm9vdC01LjguMC9zY3JpcHRzL0pTUm9vdENvcmUuanM/MmQmaGlzdCZtb3JlMmRcIlxuICAgICAgICA+PC9zY3JpcHQ+XG4gICAgICA8L0hlYWQ+XG4gICAgICA8U3R5bGVkTGF5b3V0PlxuICAgICAgICA8U3R5bGVkSGVhZGVyPlxuICAgICAgICAgIDxUb29sdGlwIHRpdGxlPVwiQmFjayB0byBtYWluIHBhZ2VcIiBwbGFjZW1lbnQ9XCJib3R0b21MZWZ0XCI+XG4gICAgICAgICAgICA8U3R5bGVkTG9nb0Rpdj5cbiAgICAgICAgICAgICAgPFN0eWxlZExvZ29XcmFwcGVyIG9uQ2xpY2s9eyhlKT0+YmFja1RvTWFpblBhZ2UoZSl9PlxuICAgICAgICAgICAgICAgIDxTdHlsZWRMb2dvIHNyYz1cIi4vaW1hZ2VzL0NNU2xvZ29fd2hpdGVfcmVkX25vbGFiZWxfMTAyNF9NYXkyMDE0LnBuZ1wiIC8+XG4gICAgICAgICAgICAgIDwvU3R5bGVkTG9nb1dyYXBwZXI+XG4gICAgICAgICAgICA8L1N0eWxlZExvZ29EaXY+XG4gICAgICAgICAgPC9Ub29sdGlwPlxuICAgICAgICAgIDxIZWFkZXJcbiAgICAgICAgICAgIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkPXtpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZH1cbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L1N0eWxlZEhlYWRlcj5cbiAgICAgICAgPENvbnRlbnRTd2l0Y2hpbmcgLz5cbiAgICAgIDwvU3R5bGVkTGF5b3V0PlxuICAgIDwvU3R5bGVkRGl2PlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgSW5kZXg7XG4iLCJleHBvcnQgaW50ZXJmYWNlIFdvcnNrYXBhY2VzUHJvcHMge1xuICBsYWJlbDogc3RyaW5nO1xuICB3b3Jrc3BhY2VzOiBhbnk7XG59XG5cbmV4cG9ydCBjb25zdCBzdW1tYXJpZXNXb3Jrc3BhY2UgPSBbXG4gIHtcbiAgICBsYWJlbDogJ1N1bW1hcnknLFxuICAgIGZvbGRlcnNQYXRoOiBbJ1N1bW1hcnknXSxcbiAgfSxcbiAgLy8ge1xuICAvLyAgIGxhYmVsOiAnUmVwb3J0cycsXG4gIC8vICAgZm9sZGVyc1BhdGg6IFtdXG4gIC8vIH0sXG4gIHtcbiAgICBsYWJlbDogJ1NoaWZ0JyxcbiAgICBmb2xkZXJzUGF0aDogWycwMCBTaGlmdCddLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdJbmZvJyxcbiAgICBmb2xkZXJzUGF0aDogWydJbmZvJ10sXG4gIH0sXG4gIC8vIHtcbiAgLy8gICBsYWJlbDogJ0NlcnRpZmljYXRpb24nLFxuICAvLyAgIGZvbGRlcnNQYXRoOiBbXVxuICAvLyB9LFxuICB7XG4gICAgbGFiZWw6ICdFdmVyeXRoaW5nJyxcbiAgICBmb2xkZXJzUGF0aDogW10sXG4gIH0sXG5dO1xuXG5jb25zdCB0cmlnZ2VyV29ya3NwYWNlID0gW1xuICB7XG4gICAgbGFiZWw6ICdMMVQnLFxuICAgIGZvbGRlcnNQYXRoOiBbJ0wxVCddLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdMMVQyMDE2RU1VJyxcbiAgICBmb2xkZXJzUGF0aDogWydMMVQyMDE2RU1VJ10sXG4gIH0sXG4gIHtcbiAgICBsYWJlbDogJ0wxVDIwMTYnLFxuICAgIGZvbGRlcnNQYXRoOiBbJ0wxVDIwMTYnXSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnTDFURU1VJyxcbiAgICBmb2xkZXJzUGF0aDogWydMMVRFTVUnXSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnSExUJyxcbiAgICBmb2xkZXJzUGF0aDogWydITFQnXSxcbiAgfSxcbl07XG5cbmNvbnN0IHRyYWNrZXJXb3Jrc3BhY2UgPSBbXG4gIHtcbiAgICBsYWJlbDogJ1BpeGVsUGhhc2UxJyxcbiAgICBmb2xkZXJzUGF0aDogWydQaXhlbFBoYXNlMSddLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdQaXhlbCcsXG4gICAgZm9sZGVyc1BhdGg6IFsnUGl4ZWwnXSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnU2lTdHJpcCcsXG4gICAgZm9sZGVyc1BhdGg6IFsnU2lTdHJpcCcsICdUcmFja2luZyddLFxuICB9LFxuXTtcblxuY29uc3QgY2Fsb3JpbWV0ZXJzV29ya3NwYWNlID0gW1xuICB7XG4gICAgbGFiZWw6ICdFY2FsJyxcbiAgICBmb2xkZXJzUGF0aDogWydFY2FsJywgJ0VjYWxCYXJyZWwnLCAnRWNhbEVuZGNhcCcsICdFY2FsQ2FsaWJyYXRpb24nXSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnRWNhbFByZXNob3dlcicsXG4gICAgZm9sZGVyc1BhdGg6IFsnRWNhbFByZXNob3dlciddLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdIQ0FMJyxcbiAgICBmb2xkZXJzUGF0aDogWydIY2FsJywgJ0hjYWwyJ10sXG4gIH0sXG4gIHtcbiAgICBsYWJlbDogJ0hDQUxjYWxpYicsXG4gICAgZm9sZGVyc1BhdGg6IFsnSGNhbENhbGliJ10sXG4gIH0sXG4gIHtcbiAgICBsYWJlbDogJ0Nhc3RvcicsXG4gICAgZm9sZGVyc1BhdGg6IFsnQ2FzdG9yJ10sXG4gIH0sXG5dO1xuXG5jb25zdCBtb3Vuc1dvcmtzcGFjZSA9IFtcbiAge1xuICAgIGxhYmVsOiAnQ1NDJyxcbiAgICBmb2xkZXJzUGF0aDogWydDU0MnXSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnRFQnLFxuICAgIGZvbGRlcnNQYXRoOiBbJ0RUJ10sXG4gIH0sXG4gIHtcbiAgICBsYWJlbDogJ1JQQycsXG4gICAgZm9sZGVyc1BhdGg6IFsnUlBDJ10sXG4gIH0sXG5dO1xuXG5jb25zdCBjdHRwc1dvcnNwYWNlID0gW1xuICB7XG4gICAgbGFiZWw6ICdUcmFja2luZ1N0cmlwJyxcbiAgICBmb2xkZXJzUGF0aDogW1xuICAgICAgJ0NUUFBTL1RyYWNraW5nU3RyaXAnLFxuICAgICAgJ0NUUFBTL2NvbW1vbicsXG4gICAgICAnQ1RQUFMvVHJhY2tpbmdTdHJpcC9MYXlvdXRzJyxcbiAgICBdLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdUcmFja2luZ1BpeGVsJyxcbiAgICBmb2xkZXJzUGF0aDogW1xuICAgICAgJ0NUUFBTL1RyYWNraW5nUGl4ZWwnLFxuICAgICAgJ0NUUFBTL2NvbW1vbicsXG4gICAgICAnQ1RQUFMvVHJhY2tpbmdQaXhlbC9MYXlvdXRzJyxcbiAgICBdLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdUaW1pbmdEaWFtb25kJyxcbiAgICBmb2xkZXJzUGF0aDogW1xuICAgICAgJ0NUUFBTL1RpbWluZ0RpYW1vbmQnLFxuICAgICAgJ0NUUFBTL2NvbW1vbicsXG4gICAgICAnQ1RQUFMvVGltaW5nRGlhbW9uZC9MYXlvdXRzJyxcbiAgICBdLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdUaW1pbmdGYXN0U2lsaWNvbicsXG4gICAgZm9sZGVyc1BhdGg6IFtcbiAgICAgICdDVFBQUy9UaW1pbmdGYXN0U2lsaWNvbicsXG4gICAgICAnQ1RQUFMvY29tbW9uJyxcbiAgICAgICdDVFBQUy9UaW1pbmdGYXN0U2lsaWNvbi9MYXlvdXRzJyxcbiAgICBdLFxuICB9LFxuXTtcblxuZXhwb3J0IGNvbnN0IHdvcmtzcGFjZXMgPSBbXG4gIHtcbiAgICBsYWJlbDogJ1N1bW1hcmllcycsXG4gICAgd29ya3NwYWNlczogc3VtbWFyaWVzV29ya3NwYWNlLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdUcmlnZ2VyJyxcbiAgICB3b3Jrc3BhY2VzOiB0cmlnZ2VyV29ya3NwYWNlLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdUcmFja2VyJyxcbiAgICB3b3Jrc3BhY2VzOiB0cmFja2VyV29ya3NwYWNlLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdDYWxvcmltZXRlcnMnLFxuICAgIHdvcmtzcGFjZXM6IGNhbG9yaW1ldGVyc1dvcmtzcGFjZSxcbiAgfSxcbiAge1xuICAgIGxhYmVsOiAnTXVvbnMnLFxuICAgIHdvcmtzcGFjZXM6IG1vdW5zV29ya3NwYWNlLFxuICB9LFxuICB7XG4gICAgbGFiZWw6ICdDVFBQUycsXG4gICAgd29ya3NwYWNlczogY3R0cHNXb3JzcGFjZSxcbiAgfSxcbl07XG4iXSwic291cmNlUm9vdCI6IiJ9
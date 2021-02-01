webpackHotUpdate_N_E("pages/index",{

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
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
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




var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/navigation/freeSearchResultModal.tsx",
    _this = undefined,
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
        lineNumber: 103,
        columnNumber: 9
      }
    }, "Close"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      key: "OK",
      onClick: onOk,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 111,
        columnNumber: 9
      }
    }, "OK")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
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
      lineNumber: 118,
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
      lineNumber: 126,
      columnNumber: 11
    }
  }), searching ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 134,
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
      lineNumber: 135,
      columnNumber: 15
    }
  })) : __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 143,
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

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbC50c3giXSwibmFtZXMiOlsib3Blbl9hX25ld190YWIiLCJxdWVyeSIsIndpbmRvdyIsIm9wZW4iLCJTZWFyY2hNb2RhbCIsInNldE1vZGFsU3RhdGUiLCJtb2RhbFN0YXRlIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2V0U2VhcmNoRGF0YXNldE5hbWUiLCJzZXRTZWFyY2hSdW5OdW1iZXIiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwidXNlU3RhdGUiLCJkYXRhc2V0TmFtZSIsInNldERhdGFzZXROYW1lIiwib3BlblJ1bkluTmV3VGFiIiwidG9nZ2xlUnVuSW5OZXdUYWIiLCJydW4iLCJydW5fbnVtYmVyIiwicnVuTnVtYmVyIiwic2V0UnVuTnVtYmVyIiwidXNlRWZmZWN0Iiwib25DbG9zaW5nIiwic2VhcmNoSGFuZGxlciIsIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwib25PayIsInBhcmFtcyIsImZvcm0iLCJnZXRGaWVsZHNWYWx1ZSIsIm5ld190YWJfcXVlcnlfcGFyYW1zIiwicXMiLCJzdHJpbmdpZnkiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjdXJyZW50X3Jvb3QiLCJsb2NhdGlvbiIsImhyZWYiLCJzcGxpdCIsInN1Ym1pdCIsIkZvcm0iLCJ1c2VGb3JtIiwidGhlbWUiLCJjb2xvcnMiLCJzZWNvbmRhcnkiLCJtYWluIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtB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QUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBa0JHN0IsVUFBVSxJQUNULG1FQUNFLE1BQUMsNkNBQUQ7QUFDRSw2QkFBeUIsRUFBRUMsaUJBRDdCO0FBRUUsK0JBQTJCLEVBQUVDLG1CQUYvQjtBQUdFLHNCQUFrQixFQUFFUSxXQUh0QjtBQUlFLG9CQUFnQixFQUFFTSxTQUpwQjtBQUtFLFdBQU8sRUFBRUssaUJBTFg7QUFNRSxRQUFJLEVBQUMsS0FOUDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsRUFTRSxNQUFDLDJEQUFEO0FBQ0UsUUFBSSxFQUFFVSxJQURSO0FBRUUsZ0JBQVksRUFBRXJCLFdBRmhCO0FBR0UsY0FBVSxFQUFFTSxTQUhkO0FBSUUscUJBQWlCLEVBQUVILGlCQUpyQjtBQUtFLG1CQUFlLEVBQUVELGVBTG5CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFURixFQWdCR2MsU0FBUyxHQUNSLE1BQUMsZ0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0VBQUQ7QUFDRSxXQUFPLEVBQUVOLGFBRFg7QUFFRSxhQUFTLEVBQUVPLFNBRmI7QUFHRSxtQkFBZSxFQUFFRixlQUhuQjtBQUlFLFVBQU0sRUFBRUcsTUFKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEUSxHQVVSLE1BQUMsZ0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQTFCSixDQW5CSixDQURGO0FBb0RELENBckhNOztHQUFNOUIsVztVQVFJUSxxRCxFQWlDMkNrQiwwRCxFQXNCM0NrQix5Q0FBSSxDQUFDQyxPOzs7S0EvRFQ3QyxXIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjE1OGM4YjNhZDNmZGJjMDkxNWVkLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHFzIGZyb20gJ3FzJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQge1xyXG4gIFN0eWxlZE1vZGFsLFxyXG4gIFJlc3VsdHNXcmFwcGVyLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IFNlbGVjdGVkRGF0YSB9IGZyb20gJy4vc2VsZWN0ZWREYXRhJztcclxuaW1wb3J0IE5hdiBmcm9tICcuLi9OYXYnO1xyXG5pbXBvcnQgeyBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5cclxuaW50ZXJmYWNlIEZyZWVTZWFjcmhNb2RhbFByb3BzIHtcclxuICBzZXRNb2RhbFN0YXRlKHN0YXRlOiBib29sZWFuKTogdm9pZDtcclxuICBtb2RhbFN0YXRlOiBib29sZWFuO1xyXG4gIHNlYXJjaF9ydW5fbnVtYmVyOiB1bmRlZmluZWQgfCBzdHJpbmc7XHJcbiAgc2VhcmNoX2RhdGFzZXRfbmFtZTogc3RyaW5nIHwgdW5kZWZpbmVkO1xyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lKGRhdGFzZXRfbmFtZTogYW55KTogdm9pZDtcclxuICBzZXRTZWFyY2hSdW5OdW1iZXIocnVuX251bWJlcjogc3RyaW5nKTogdm9pZDtcclxufVxyXG5cclxuY29uc3Qgb3Blbl9hX25ld190YWIgPSAocXVlcnk6IHN0cmluZykgPT4ge1xyXG4gIHdpbmRvdy5vcGVuKHF1ZXJ5LCAnX2JsYW5rJyk7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgU2VhcmNoTW9kYWwgPSAoe1xyXG4gIHNldE1vZGFsU3RhdGUsXHJcbiAgbW9kYWxTdGF0ZSxcclxuICBzZWFyY2hfcnVuX251bWJlcixcclxuICBzZWFyY2hfZGF0YXNldF9uYW1lLFxyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lLFxyXG4gIHNldFNlYXJjaFJ1bk51bWJlcixcclxufTogRnJlZVNlYWNyaE1vZGFsUHJvcHMpID0+IHtcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcblxyXG4gIGNvbnN0IFtkYXRhc2V0TmFtZSwgc2V0RGF0YXNldE5hbWVdID0gdXNlU3RhdGUoZGF0YXNldCk7XHJcbiAgY29uc3QgW29wZW5SdW5Jbk5ld1RhYiwgdG9nZ2xlUnVuSW5OZXdUYWJdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgY29uc3QgW3J1bk51bWJlciwgc2V0UnVuTnVtYmVyXSA9IHVzZVN0YXRlPHN0cmluZz4ocnVuKTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0KTtcclxuICAgIHNldFJ1bk51bWJlcihydW4pO1xyXG4gIH0sIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXJdKTtcclxuXHJcbiAgY29uc3Qgb25DbG9zaW5nID0gKCkgPT4ge1xyXG4gICAgc2V0TW9kYWxTdGF0ZShmYWxzZSk7XHJcbiAgfTtcclxuXHJcbiAgY29uc3Qgc2VhcmNoSGFuZGxlciA9IChydW5fbnVtYmVyOiBzdHJpbmcsIGRhdGFzZXRfbmFtZTogc3RyaW5nKSA9PiB7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0X25hbWUpO1xyXG4gICAgc2V0UnVuTnVtYmVyKHJ1bl9udW1iZXIpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IG5hdmlnYXRpb25IYW5kbGVyID0gKFxyXG4gICAgc2VhcmNoX2J5X3J1bl9udW1iZXI6IHN0cmluZyxcclxuICAgIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZ1xyXG4gICkgPT4ge1xyXG4gICAgc2V0U2VhcmNoUnVuTnVtYmVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyKTtcclxuICAgIHNldFNlYXJjaERhdGFzZXROYW1lKHNlYXJjaF9ieV9kYXRhc2V0X25hbWUpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBzZWFyY2hpbmcsIGlzTG9hZGluZywgZXJyb3JzIH0gPSB1c2VTZWFyY2goXHJcbiAgICBzZWFyY2hfcnVuX251bWJlcixcclxuICAgIHNlYXJjaF9kYXRhc2V0X25hbWVcclxuICApO1xyXG5cclxuICBjb25zdCBvbk9rID0gYXN5bmMgKCkgPT4ge1xyXG4gICAgaWYgKG9wZW5SdW5Jbk5ld1RhYikge1xyXG4gICAgICBjb25zdCBwYXJhbXMgPSBmb3JtLmdldEZpZWxkc1ZhbHVlKCk7XHJcbiAgICAgIGNvbnN0IG5ld190YWJfcXVlcnlfcGFyYW1zID0gcXMuc3RyaW5naWZ5KFxyXG4gICAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhwYXJhbXMsIHF1ZXJ5KVxyXG4gICAgICApO1xyXG4gICAgICAvL3Jvb3QgdXJsIGlzIGVuZHMgd2l0aCBmaXJzdCAnPycuIEkgY2FuJ3QgdXNlIGp1c3Qgcm9vdCB1cmwgZnJvbSBjb25maWcuY29uZmlnLCBiZWNhdXNlXHJcbiAgICAgIC8vaW4gZGV2IGVudiBpdCB1c2UgbG9jYWxob3N0OjgwODEvZHFtL2RldiAodGhpcyBpcyBvbGQgYmFja2VuZCB1cmwgZnJvbSB3aGVyZSBJJ20gZ2V0dGluZyBkYXRhKSxcclxuICAgICAgLy9idXQgSSBuZWVkIGxvY2FsaG9zdDozMDAwXHJcbiAgICAgIGNvbnN0IGN1cnJlbnRfcm9vdCA9IHdpbmRvdy5sb2NhdGlvbi5ocmVmLnNwbGl0KCcvPycpWzBdO1xyXG4gICAgICBvcGVuX2FfbmV3X3RhYihgJHtjdXJyZW50X3Jvb3R9Lz8ke25ld190YWJfcXVlcnlfcGFyYW1zfWApO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgYXdhaXQgZm9ybS5zdWJtaXQoKTtcclxuICAgIH1cclxuICAgIG9uQ2xvc2luZygpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZE1vZGFsXHJcbiAgICAgIHRpdGxlPVwiU2VhcmNoIGRhdGFcIlxyXG4gICAgICB2aXNpYmxlPXttb2RhbFN0YXRlfVxyXG4gICAgICBvbkNhbmNlbD17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgIGZvb3Rlcj17W1xyXG4gICAgICAgIDxTdHlsZWRCdXR0b25cclxuICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XHJcbiAgICAgICAgICBiYWNrZ3JvdW5kPVwid2hpdGVcIlxyXG4gICAgICAgICAga2V5PVwiQ2xvc2VcIlxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgQ2xvc2VcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgICAgPFN0eWxlZEJ1dHRvbiBrZXk9XCJPS1wiIG9uQ2xpY2s9e29uT2t9PlxyXG4gICAgICAgICAgT0tcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgIF19XHJcbiAgICA+XHJcbiAgICAgIHttb2RhbFN0YXRlICYmIChcclxuICAgICAgICA8PlxyXG4gICAgICAgICAgPE5hdlxyXG4gICAgICAgICAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPXtzZWFyY2hfcnVuX251bWJlcn1cclxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtzZWFyY2hfZGF0YXNldF9uYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0RGF0YXNldE5hbWU9e2RhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0UnVuTnVtYmVyPXtydW5OdW1iZXJ9XHJcbiAgICAgICAgICAgIGhhbmRsZXI9e25hdmlnYXRpb25IYW5kbGVyfVxyXG4gICAgICAgICAgICB0eXBlPVwidG9wXCJcclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8U2VsZWN0ZWREYXRhXHJcbiAgICAgICAgICAgIGZvcm09e2Zvcm19XHJcbiAgICAgICAgICAgIGRhdGFzZXRfbmFtZT17ZGF0YXNldE5hbWV9XHJcbiAgICAgICAgICAgIHJ1bl9udW1iZXI9e3J1bk51bWJlcn1cclxuICAgICAgICAgICAgdG9nZ2xlUnVuSW5OZXdUYWI9e3RvZ2dsZVJ1bkluTmV3VGFifVxyXG4gICAgICAgICAgICBvcGVuUnVuSW5OZXdUYWI9e29wZW5SdW5Jbk5ld1RhYn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICB7c2VhcmNoaW5nID8gKFxyXG4gICAgICAgICAgICA8UmVzdWx0c1dyYXBwZXI+XHJcbiAgICAgICAgICAgICAgPFNlYXJjaFJlc3VsdHNcclxuICAgICAgICAgICAgICAgIGhhbmRsZXI9e3NlYXJjaEhhbmRsZXJ9XHJcbiAgICAgICAgICAgICAgICBpc0xvYWRpbmc9e2lzTG9hZGluZ31cclxuICAgICAgICAgICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxyXG4gICAgICAgICAgICAgICAgZXJyb3JzPXtlcnJvcnN9XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9SZXN1bHRzV3JhcHBlcj5cclxuICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgIDxSZXN1bHRzV3JhcHBlciAvPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICA8Lz5cclxuICAgICAgKX1cclxuICAgIDwvU3R5bGVkTW9kYWw+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==
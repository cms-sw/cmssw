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

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbC50c3giXSwibmFtZXMiOlsib3Blbl9hX25ld190YWIiLCJxdWVyeSIsIndpbmRvdyIsIm9wZW4iLCJTZWFyY2hNb2RhbCIsInNldE1vZGFsU3RhdGUiLCJtb2RhbFN0YXRlIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2V0U2VhcmNoRGF0YXNldE5hbWUiLCJzZXRTZWFyY2hSdW5OdW1iZXIiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwidXNlU3RhdGUiLCJkYXRhc2V0TmFtZSIsInNldERhdGFzZXROYW1lIiwib3BlblJ1bkluTmV3VGFiIiwidG9nZ2xlUnVuSW5OZXdUYWIiLCJydW4iLCJydW5fbnVtYmVyIiwicnVuTnVtYmVyIiwic2V0UnVuTnVtYmVyIiwidXNlRWZmZWN0Iiwib25DbG9zaW5nIiwic2VhcmNoSGFuZGxlciIsIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwib25PayIsInBhcmFtcyIsImZvcm0iLCJnZXRGaWVsZHNWYWx1ZSIsIm5ld190YWJfcXVlcnlfcGFyYW1zIiwicXMiLCJzdHJpbmdpZnkiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjdXJyZW50X3Jvb3QiLCJsb2NhdGlvbiIsImhyZWYiLCJzcGxpdCIsInN1Ym1pdCIsIkZvcm0iLCJ1c2VGb3JtIiwidGhlbWUiLCJjb2xvcnMiLCJzZWNvbmRhcnkiLCJtYWluIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBQ0E7QUFFQTtB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QUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBa0JHN0IsVUFBVSxJQUNULG1FQUNFLE1BQUMsNkNBQUQ7QUFDRSw2QkFBeUIsRUFBRUMsaUJBRDdCO0FBRUUsK0JBQTJCLEVBQUVDLG1CQUYvQjtBQUdFLHNCQUFrQixFQUFFUSxXQUh0QjtBQUlFLG9CQUFnQixFQUFFTSxTQUpwQjtBQUtFLFdBQU8sRUFBRUssaUJBTFg7QUFNRSxRQUFJLEVBQUMsS0FOUDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsRUFTRSxNQUFDLDJEQUFEO0FBQ0UsUUFBSSxFQUFFVSxJQURSO0FBRUUsZ0JBQVksRUFBRXJCLFdBRmhCO0FBR0UsY0FBVSxFQUFFTSxTQUhkO0FBSUUscUJBQWlCLEVBQUVILGlCQUpyQjtBQUtFLG1CQUFlLEVBQUVELGVBTG5CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFURixFQWdCR2MsU0FBUyxHQUNSLE1BQUMsZ0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0VBQUQ7QUFDRSxXQUFPLEVBQUVOLGFBRFg7QUFFRSxhQUFTLEVBQUVPLFNBRmI7QUFHRSxtQkFBZSxFQUFFRixlQUhuQjtBQUlFLFVBQU0sRUFBRUcsTUFKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEUSxHQVVSLE1BQUMsZ0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQTFCSixDQW5CSixDQURGO0FBb0RELENBckhNOztHQUFNOUIsVztVQVFJUSxxRCxFQWlDMkNrQiwwRCxFQXNCM0NrQix5Q0FBSSxDQUFDQyxPOzs7S0EvRFQ3QyxXIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjYxMmIzNTM3NGNkNjc3NmExYmE5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHFzIGZyb20gJ3FzJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQge1xyXG4gIFN0eWxlZE1vZGFsLFxyXG4gIFJlc3VsdHNXcmFwcGVyLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IFNlbGVjdGVkRGF0YSB9IGZyb20gJy4vc2VsZWN0ZWREYXRhJztcclxuaW1wb3J0IE5hdiBmcm9tICcuLi9OYXYnO1xyXG5pbXBvcnQgeyBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge3N0b3JlfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnXHJcbmludGVyZmFjZSBGcmVlU2VhY3JoTW9kYWxQcm9wcyB7XHJcbiAgc2V0TW9kYWxTdGF0ZShzdGF0ZTogYm9vbGVhbik6IHZvaWQ7XHJcbiAgbW9kYWxTdGF0ZTogYm9vbGVhbjtcclxuICBzZWFyY2hfcnVuX251bWJlcjogdW5kZWZpbmVkIHwgc3RyaW5nO1xyXG4gIHNlYXJjaF9kYXRhc2V0X25hbWU6IHN0cmluZyB8IHVuZGVmaW5lZDtcclxuICBzZXRTZWFyY2hEYXRhc2V0TmFtZShkYXRhc2V0X25hbWU6IGFueSk6IHZvaWQ7XHJcbiAgc2V0U2VhcmNoUnVuTnVtYmVyKHJ1bl9udW1iZXI6IHN0cmluZyk6IHZvaWQ7XHJcbn1cclxuXHJcbmNvbnN0IG9wZW5fYV9uZXdfdGFiID0gKHF1ZXJ5OiBzdHJpbmcpID0+IHtcclxuICB3aW5kb3cub3BlbihxdWVyeSwgJ19ibGFuaycpO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IFNlYXJjaE1vZGFsID0gKHtcclxuICBzZXRNb2RhbFN0YXRlLFxyXG4gIG1vZGFsU3RhdGUsXHJcbiAgc2VhcmNoX3J1bl9udW1iZXIsXHJcbiAgc2VhcmNoX2RhdGFzZXRfbmFtZSxcclxuICBzZXRTZWFyY2hEYXRhc2V0TmFtZSxcclxuICBzZXRTZWFyY2hSdW5OdW1iZXIsXHJcbn06IEZyZWVTZWFjcmhNb2RhbFByb3BzKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgY29uc3QgZGF0YXNldCA9IHF1ZXJ5LmRhdGFzZXRfbmFtZSA/IHF1ZXJ5LmRhdGFzZXRfbmFtZSA6ICcnO1xyXG5cclxuICBjb25zdCBbZGF0YXNldE5hbWUsIHNldERhdGFzZXROYW1lXSA9IHVzZVN0YXRlKGRhdGFzZXQpO1xyXG4gIGNvbnN0IFtvcGVuUnVuSW5OZXdUYWIsIHRvZ2dsZVJ1bkluTmV3VGFiXSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBydW4gPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xyXG4gIGNvbnN0IFtydW5OdW1iZXIsIHNldFJ1bk51bWJlcl0gPSB1c2VTdGF0ZTxzdHJpbmc+KHJ1bik7XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBydW4gPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xyXG4gICAgY29uc3QgZGF0YXNldCA9IHF1ZXJ5LmRhdGFzZXRfbmFtZSA/IHF1ZXJ5LmRhdGFzZXRfbmFtZSA6ICcnO1xyXG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldCk7XHJcbiAgICBzZXRSdW5OdW1iZXIocnVuKTtcclxuICB9LCBbcXVlcnkuZGF0YXNldF9uYW1lLCBxdWVyeS5ydW5fbnVtYmVyXSk7XHJcblxyXG4gIGNvbnN0IG9uQ2xvc2luZyA9ICgpID0+IHtcclxuICAgIHNldE1vZGFsU3RhdGUoZmFsc2UpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IHNlYXJjaEhhbmRsZXIgPSAocnVuX251bWJlcjogc3RyaW5nLCBkYXRhc2V0X25hbWU6IHN0cmluZykgPT4ge1xyXG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldF9uYW1lKTtcclxuICAgIHNldFJ1bk51bWJlcihydW5fbnVtYmVyKTtcclxuICB9O1xyXG5cclxuICBjb25zdCBuYXZpZ2F0aW9uSGFuZGxlciA9IChcclxuICAgIHNlYXJjaF9ieV9ydW5fbnVtYmVyOiBzdHJpbmcsXHJcbiAgICBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmdcclxuICApID0+IHtcclxuICAgIHNldFNlYXJjaFJ1bk51bWJlcihzZWFyY2hfYnlfcnVuX251bWJlcik7XHJcbiAgICBzZXRTZWFyY2hEYXRhc2V0TmFtZShzZWFyY2hfYnlfZGF0YXNldF9uYW1lKTtcclxuICB9O1xyXG5cclxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgc2VhcmNoaW5nLCBpc0xvYWRpbmcsIGVycm9ycyB9ID0gdXNlU2VhcmNoKFxyXG4gICAgc2VhcmNoX3J1bl9udW1iZXIsXHJcbiAgICBzZWFyY2hfZGF0YXNldF9uYW1lXHJcbiAgKTtcclxuXHJcbiAgY29uc3Qgb25PayA9IGFzeW5jICgpID0+IHtcclxuICAgIGlmIChvcGVuUnVuSW5OZXdUYWIpIHtcclxuICAgICAgY29uc3QgcGFyYW1zID0gZm9ybS5nZXRGaWVsZHNWYWx1ZSgpO1xyXG4gICAgICBjb25zdCBuZXdfdGFiX3F1ZXJ5X3BhcmFtcyA9IHFzLnN0cmluZ2lmeShcclxuICAgICAgICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMocGFyYW1zLCBxdWVyeSlcclxuICAgICAgKTtcclxuICAgICAgLy9yb290IHVybCBpcyBlbmRzIHdpdGggZmlyc3QgJz8nLiBJIGNhbid0IHVzZSBqdXN0IHJvb3QgdXJsIGZyb20gY29uZmlnLmNvbmZpZywgYmVjYXVzZVxyXG4gICAgICAvL2luIGRldiBlbnYgaXQgdXNlIGxvY2FsaG9zdDo4MDgxL2RxbS9kZXYgKHRoaXMgaXMgb2xkIGJhY2tlbmQgdXJsIGZyb20gd2hlcmUgSSdtIGdldHRpbmcgZGF0YSksXHJcbiAgICAgIC8vYnV0IEkgbmVlZCBsb2NhbGhvc3Q6MzAwMFxyXG4gICAgICBjb25zdCBjdXJyZW50X3Jvb3QgPSB3aW5kb3cubG9jYXRpb24uaHJlZi5zcGxpdCgnLz8nKVswXTtcclxuICAgICAgb3Blbl9hX25ld190YWIoYCR7Y3VycmVudF9yb290fS8/JHtuZXdfdGFiX3F1ZXJ5X3BhcmFtc31gKTtcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgIGF3YWl0IGZvcm0uc3VibWl0KCk7XHJcbiAgICB9XHJcbiAgICBvbkNsb3NpbmcoKTtcclxuICB9O1xyXG5cclxuICBjb25zdCBbZm9ybV0gPSBGb3JtLnVzZUZvcm0oKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRNb2RhbFxyXG4gICAgICB0aXRsZT1cIlNlYXJjaCBkYXRhXCJcclxuICAgICAgdmlzaWJsZT17bW9kYWxTdGF0ZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IG9uQ2xvc2luZygpfVxyXG4gICAgICBmb290ZXI9e1tcclxuICAgICAgICA8U3R5bGVkQnV0dG9uXHJcbiAgICAgICAgICBjb2xvcj17dGhlbWUuY29sb3JzLnNlY29uZGFyeS5tYWlufVxyXG4gICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcclxuICAgICAgICAgIGtleT1cIkNsb3NlXCJcclxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IG9uQ2xvc2luZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIENsb3NlXHJcbiAgICAgICAgPC9TdHlsZWRCdXR0b24+LFxyXG4gICAgICAgIDxTdHlsZWRCdXR0b24ga2V5PVwiT0tcIiBvbkNsaWNrPXtvbk9rfT5cclxuICAgICAgICAgIE9LXHJcbiAgICAgICAgPC9TdHlsZWRCdXR0b24+LFxyXG4gICAgICBdfVxyXG4gICAgPlxyXG4gICAgICB7bW9kYWxTdGF0ZSAmJiAoXHJcbiAgICAgICAgPD5cclxuICAgICAgICAgIDxOYXZcclxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj17c2VhcmNoX3J1bl9udW1iZXJ9XHJcbiAgICAgICAgICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT17c2VhcmNoX2RhdGFzZXRfbmFtZX1cclxuICAgICAgICAgICAgZGVmYXVsdERhdGFzZXROYW1lPXtkYXRhc2V0TmFtZX1cclxuICAgICAgICAgICAgZGVmYXVsdFJ1bk51bWJlcj17cnVuTnVtYmVyfVxyXG4gICAgICAgICAgICBoYW5kbGVyPXtuYXZpZ2F0aW9uSGFuZGxlcn1cclxuICAgICAgICAgICAgdHlwZT1cInRvcFwiXHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgICAgPFNlbGVjdGVkRGF0YVxyXG4gICAgICAgICAgICBmb3JtPXtmb3JtfVxyXG4gICAgICAgICAgICBkYXRhc2V0X25hbWU9e2RhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICBydW5fbnVtYmVyPXtydW5OdW1iZXJ9XHJcbiAgICAgICAgICAgIHRvZ2dsZVJ1bkluTmV3VGFiPXt0b2dnbGVSdW5Jbk5ld1RhYn1cclxuICAgICAgICAgICAgb3BlblJ1bkluTmV3VGFiPXtvcGVuUnVuSW5OZXdUYWJ9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgICAge3NlYXJjaGluZyA/IChcclxuICAgICAgICAgICAgPFJlc3VsdHNXcmFwcGVyPlxyXG4gICAgICAgICAgICAgIDxTZWFyY2hSZXN1bHRzXHJcbiAgICAgICAgICAgICAgICBoYW5kbGVyPXtzZWFyY2hIYW5kbGVyfVxyXG4gICAgICAgICAgICAgICAgaXNMb2FkaW5nPXtpc0xvYWRpbmd9XHJcbiAgICAgICAgICAgICAgICByZXN1bHRzX2dyb3VwZWQ9e3Jlc3VsdHNfZ3JvdXBlZH1cclxuICAgICAgICAgICAgICAgIGVycm9ycz17ZXJyb3JzfVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvUmVzdWx0c1dyYXBwZXI+XHJcbiAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICA8UmVzdWx0c1dyYXBwZXIgLz5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC8+XHJcbiAgICAgICl9XHJcbiAgICA8L1N0eWxlZE1vZGFsPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
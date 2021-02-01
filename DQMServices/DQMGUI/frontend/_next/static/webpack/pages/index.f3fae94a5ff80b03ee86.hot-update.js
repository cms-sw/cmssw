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

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3___default.a.useContext(store),
      set_updated_by_not_older_than = _React$useContext.set_updated_by_not_older_than,
      update = _React$useContext.update,
      set_update = _React$useContext.set_update;

  console.log(update);

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
        lineNumber: 111,
        columnNumber: 9
      }
    }, "Close"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      key: "OK",
      onClick: onOk,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 119,
        columnNumber: 9
      }
    }, "OK")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 106,
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
      lineNumber: 126,
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
      lineNumber: 134,
      columnNumber: 11
    }
  }), searching ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 142,
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
      lineNumber: 143,
      columnNumber: 15
    }
  })) : __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 151,
      columnNumber: 13
    }
  })));
};

_s(SearchModal, "hoXvPVRrv6LjAVi68uX+kDmwPdw=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbC50c3giXSwibmFtZXMiOlsib3Blbl9hX25ld190YWIiLCJxdWVyeSIsIndpbmRvdyIsIm9wZW4iLCJTZWFyY2hNb2RhbCIsInNldE1vZGFsU3RhdGUiLCJtb2RhbFN0YXRlIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2V0U2VhcmNoRGF0YXNldE5hbWUiLCJzZXRTZWFyY2hSdW5OdW1iZXIiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwiUmVhY3QiLCJ1c2VDb250ZXh0Iiwic3RvcmUiLCJzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInVwZGF0ZSIsInNldF91cGRhdGUiLCJjb25zb2xlIiwibG9nIiwidXNlU3RhdGUiLCJkYXRhc2V0TmFtZSIsInNldERhdGFzZXROYW1lIiwib3BlblJ1bkluTmV3VGFiIiwidG9nZ2xlUnVuSW5OZXdUYWIiLCJydW4iLCJydW5fbnVtYmVyIiwicnVuTnVtYmVyIiwic2V0UnVuTnVtYmVyIiwidXNlRWZmZWN0Iiwib25DbG9zaW5nIiwic2VhcmNoSGFuZGxlciIsIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwib25PayIsInBhcmFtcyIsImZvcm0iLCJnZXRGaWVsZHNWYWx1ZSIsIm5ld190YWJfcXVlcnlfcGFyYW1zIiwicXMiLCJzdHJpbmdpZnkiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjdXJyZW50X3Jvb3QiLCJsb2NhdGlvbiIsImhyZWYiLCJzcGxpdCIsInN1Ym1pdCIsIkZvcm0iLCJ1c2VGb3JtIiwidGhlbWUiLCJjb2xvcnMi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LElBREYsQ0FEUSxHQVVSLE1BQUMsZ0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQTFCSixDQW5CSixDQURGO0FBb0RELENBNUhNOztHQUFNdEMsVztVQVFJUSxxRCxFQXdDMkMwQiwwRCxFQXNCM0NrQix5Q0FBSSxDQUFDQyxPOzs7S0F0RVRyRCxXIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmYzZmFlOTRhNWZmODBiMDNlZTg2LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHFzIGZyb20gJ3FzJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQge1xyXG4gIFN0eWxlZE1vZGFsLFxyXG4gIFJlc3VsdHNXcmFwcGVyLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IFNlbGVjdGVkRGF0YSB9IGZyb20gJy4vc2VsZWN0ZWREYXRhJztcclxuaW1wb3J0IE5hdiBmcm9tICcuLi9OYXYnO1xyXG5pbXBvcnQgeyBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5cclxuaW50ZXJmYWNlIEZyZWVTZWFjcmhNb2RhbFByb3BzIHtcclxuICBzZXRNb2RhbFN0YXRlKHN0YXRlOiBib29sZWFuKTogdm9pZDtcclxuICBtb2RhbFN0YXRlOiBib29sZWFuO1xyXG4gIHNlYXJjaF9ydW5fbnVtYmVyOiB1bmRlZmluZWQgfCBzdHJpbmc7XHJcbiAgc2VhcmNoX2RhdGFzZXRfbmFtZTogc3RyaW5nIHwgdW5kZWZpbmVkO1xyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lKGRhdGFzZXRfbmFtZTogYW55KTogdm9pZDtcclxuICBzZXRTZWFyY2hSdW5OdW1iZXIocnVuX251bWJlcjogc3RyaW5nKTogdm9pZDtcclxufVxyXG5cclxuY29uc3Qgb3Blbl9hX25ld190YWIgPSAocXVlcnk6IHN0cmluZykgPT4ge1xyXG4gIHdpbmRvdy5vcGVuKHF1ZXJ5LCAnX2JsYW5rJyk7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgU2VhcmNoTW9kYWwgPSAoe1xyXG4gIHNldE1vZGFsU3RhdGUsXHJcbiAgbW9kYWxTdGF0ZSxcclxuICBzZWFyY2hfcnVuX251bWJlcixcclxuICBzZWFyY2hfZGF0YXNldF9uYW1lLFxyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lLFxyXG4gIHNldFNlYXJjaFJ1bk51bWJlcixcclxufTogRnJlZVNlYWNyaE1vZGFsUHJvcHMpID0+IHtcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcblxyXG4gIGNvbnN0IHtcclxuICAgIHNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgdXBkYXRlLFxyXG4gICAgc2V0X3VwZGF0ZSxcclxuICB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIGNvbnNvbGUubG9nKHVwZGF0ZSlcclxuICBjb25zdCBbZGF0YXNldE5hbWUsIHNldERhdGFzZXROYW1lXSA9IHVzZVN0YXRlKGRhdGFzZXQpO1xyXG4gIGNvbnN0IFtvcGVuUnVuSW5OZXdUYWIsIHRvZ2dsZVJ1bkluTmV3VGFiXSA9IHVzZVN0YXRlKGZhbHNlKTtcclxuICBjb25zdCBydW4gPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xyXG4gIGNvbnN0IFtydW5OdW1iZXIsIHNldFJ1bk51bWJlcl0gPSB1c2VTdGF0ZTxzdHJpbmc+KHJ1bik7XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBydW4gPSBxdWVyeS5ydW5fbnVtYmVyID8gcXVlcnkucnVuX251bWJlciA6ICcnO1xyXG4gICAgY29uc3QgZGF0YXNldCA9IHF1ZXJ5LmRhdGFzZXRfbmFtZSA/IHF1ZXJ5LmRhdGFzZXRfbmFtZSA6ICcnO1xyXG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldCk7XHJcbiAgICBzZXRSdW5OdW1iZXIocnVuKTtcclxuICB9LCBbcXVlcnkuZGF0YXNldF9uYW1lLCBxdWVyeS5ydW5fbnVtYmVyXSk7XHJcblxyXG4gIGNvbnN0IG9uQ2xvc2luZyA9ICgpID0+IHtcclxuICAgIHNldE1vZGFsU3RhdGUoZmFsc2UpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IHNlYXJjaEhhbmRsZXIgPSAocnVuX251bWJlcjogc3RyaW5nLCBkYXRhc2V0X25hbWU6IHN0cmluZykgPT4ge1xyXG4gICAgc2V0RGF0YXNldE5hbWUoZGF0YXNldF9uYW1lKTtcclxuICAgIHNldFJ1bk51bWJlcihydW5fbnVtYmVyKTtcclxuICB9O1xyXG5cclxuICBjb25zdCBuYXZpZ2F0aW9uSGFuZGxlciA9IChcclxuICAgIHNlYXJjaF9ieV9ydW5fbnVtYmVyOiBzdHJpbmcsXHJcbiAgICBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmdcclxuICApID0+IHtcclxuICAgIHNldFNlYXJjaFJ1bk51bWJlcihzZWFyY2hfYnlfcnVuX251bWJlcik7XHJcbiAgICBzZXRTZWFyY2hEYXRhc2V0TmFtZShzZWFyY2hfYnlfZGF0YXNldF9uYW1lKTtcclxuICB9O1xyXG5cclxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgc2VhcmNoaW5nLCBpc0xvYWRpbmcsIGVycm9ycyB9ID0gdXNlU2VhcmNoKFxyXG4gICAgc2VhcmNoX3J1bl9udW1iZXIsXHJcbiAgICBzZWFyY2hfZGF0YXNldF9uYW1lXHJcbiAgKTtcclxuXHJcbiAgY29uc3Qgb25PayA9IGFzeW5jICgpID0+IHtcclxuICAgIGlmIChvcGVuUnVuSW5OZXdUYWIpIHtcclxuICAgICAgY29uc3QgcGFyYW1zID0gZm9ybS5nZXRGaWVsZHNWYWx1ZSgpO1xyXG4gICAgICBjb25zdCBuZXdfdGFiX3F1ZXJ5X3BhcmFtcyA9IHFzLnN0cmluZ2lmeShcclxuICAgICAgICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMocGFyYW1zLCBxdWVyeSlcclxuICAgICAgKTtcclxuICAgICAgLy9yb290IHVybCBpcyBlbmRzIHdpdGggZmlyc3QgJz8nLiBJIGNhbid0IHVzZSBqdXN0IHJvb3QgdXJsIGZyb20gY29uZmlnLmNvbmZpZywgYmVjYXVzZVxyXG4gICAgICAvL2luIGRldiBlbnYgaXQgdXNlIGxvY2FsaG9zdDo4MDgxL2RxbS9kZXYgKHRoaXMgaXMgb2xkIGJhY2tlbmQgdXJsIGZyb20gd2hlcmUgSSdtIGdldHRpbmcgZGF0YSksXHJcbiAgICAgIC8vYnV0IEkgbmVlZCBsb2NhbGhvc3Q6MzAwMFxyXG4gICAgICBjb25zdCBjdXJyZW50X3Jvb3QgPSB3aW5kb3cubG9jYXRpb24uaHJlZi5zcGxpdCgnLz8nKVswXTtcclxuICAgICAgb3Blbl9hX25ld190YWIoYCR7Y3VycmVudF9yb290fS8/JHtuZXdfdGFiX3F1ZXJ5X3BhcmFtc31gKTtcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgIGF3YWl0IGZvcm0uc3VibWl0KCk7XHJcbiAgICB9XHJcbiAgICBvbkNsb3NpbmcoKTtcclxuICB9O1xyXG5cclxuICBjb25zdCBbZm9ybV0gPSBGb3JtLnVzZUZvcm0oKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRNb2RhbFxyXG4gICAgICB0aXRsZT1cIlNlYXJjaCBkYXRhXCJcclxuICAgICAgdmlzaWJsZT17bW9kYWxTdGF0ZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IG9uQ2xvc2luZygpfVxyXG4gICAgICBmb290ZXI9e1tcclxuICAgICAgICA8U3R5bGVkQnV0dG9uXHJcbiAgICAgICAgICBjb2xvcj17dGhlbWUuY29sb3JzLnNlY29uZGFyeS5tYWlufVxyXG4gICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcclxuICAgICAgICAgIGtleT1cIkNsb3NlXCJcclxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IG9uQ2xvc2luZygpfVxyXG4gICAgICAgID5cclxuICAgICAgICAgIENsb3NlXHJcbiAgICAgICAgPC9TdHlsZWRCdXR0b24+LFxyXG4gICAgICAgIDxTdHlsZWRCdXR0b24ga2V5PVwiT0tcIiBvbkNsaWNrPXtvbk9rfT5cclxuICAgICAgICAgIE9LXHJcbiAgICAgICAgPC9TdHlsZWRCdXR0b24+LFxyXG4gICAgICBdfVxyXG4gICAgPlxyXG4gICAgICB7bW9kYWxTdGF0ZSAmJiAoXHJcbiAgICAgICAgPD5cclxuICAgICAgICAgIDxOYXZcclxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj17c2VhcmNoX3J1bl9udW1iZXJ9XHJcbiAgICAgICAgICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT17c2VhcmNoX2RhdGFzZXRfbmFtZX1cclxuICAgICAgICAgICAgZGVmYXVsdERhdGFzZXROYW1lPXtkYXRhc2V0TmFtZX1cclxuICAgICAgICAgICAgZGVmYXVsdFJ1bk51bWJlcj17cnVuTnVtYmVyfVxyXG4gICAgICAgICAgICBoYW5kbGVyPXtuYXZpZ2F0aW9uSGFuZGxlcn1cclxuICAgICAgICAgICAgdHlwZT1cInRvcFwiXHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgICAgPFNlbGVjdGVkRGF0YVxyXG4gICAgICAgICAgICBmb3JtPXtmb3JtfVxyXG4gICAgICAgICAgICBkYXRhc2V0X25hbWU9e2RhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICBydW5fbnVtYmVyPXtydW5OdW1iZXJ9XHJcbiAgICAgICAgICAgIHRvZ2dsZVJ1bkluTmV3VGFiPXt0b2dnbGVSdW5Jbk5ld1RhYn1cclxuICAgICAgICAgICAgb3BlblJ1bkluTmV3VGFiPXtvcGVuUnVuSW5OZXdUYWJ9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgICAge3NlYXJjaGluZyA/IChcclxuICAgICAgICAgICAgPFJlc3VsdHNXcmFwcGVyPlxyXG4gICAgICAgICAgICAgIDxTZWFyY2hSZXN1bHRzXHJcbiAgICAgICAgICAgICAgICBoYW5kbGVyPXtzZWFyY2hIYW5kbGVyfVxyXG4gICAgICAgICAgICAgICAgaXNMb2FkaW5nPXtpc0xvYWRpbmd9XHJcbiAgICAgICAgICAgICAgICByZXN1bHRzX2dyb3VwZWQ9e3Jlc3VsdHNfZ3JvdXBlZH1cclxuICAgICAgICAgICAgICAgIGVycm9ycz17ZXJyb3JzfVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIDwvUmVzdWx0c1dyYXBwZXI+XHJcbiAgICAgICAgICApIDogKFxyXG4gICAgICAgICAgICA8UmVzdWx0c1dyYXBwZXIgLz5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC8+XHJcbiAgICAgICl9XHJcbiAgICA8L1N0eWxlZE1vZGFsPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=
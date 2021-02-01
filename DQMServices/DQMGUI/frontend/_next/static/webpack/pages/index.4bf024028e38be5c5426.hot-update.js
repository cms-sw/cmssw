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
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




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

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3___default.a.useContext(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_15__["store"]),
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbC50c3giXSwibmFtZXMiOlsib3Blbl9hX25ld190YWIiLCJxdWVyeSIsIndpbmRvdyIsIm9wZW4iLCJTZWFyY2hNb2RhbCIsInNldE1vZGFsU3RhdGUiLCJtb2RhbFN0YXRlIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2V0U2VhcmNoRGF0YXNldE5hbWUiLCJzZXRTZWFyY2hSdW5OdW1iZXIiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwiUmVhY3QiLCJ1c2VDb250ZXh0Iiwic3RvcmUiLCJzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInVwZGF0ZSIsInNldF91cGRhdGUiLCJjb25zb2xlIiwibG9nIiwidXNlU3RhdGUiLCJkYXRhc2V0TmFtZSIsInNldERhdGFzZXROYW1lIiwib3BlblJ1bkluTmV3VGFiIiwidG9nZ2xlUnVuSW5OZXdUYWIiLCJydW4iLCJydW5fbnVtYmVyIiwicnVuTnVtYmVyIiwic2V0UnVuTnVtYmVyIiwidXNlRWZmZWN0Iiwib25DbG9zaW5nIiwic2VhcmNoSGFuZGxlciIsIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwidXNlU2VhcmNoIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwib25PayIsInBhcmFtcyIsImZvcm0iLCJnZXRGaWVsZHNWYWx1ZSIsIm5ld190YWJfcXVlcnlfcGFyYW1zIiwicXMiLCJzdHJpbmdpZnkiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjdXJyZW50X3Jvb3QiLCJsb2NhdGlvbiIsImhyZWYiLCJzcGxpdCIsInN1Ym1pdCIsIkZvcm0iLCJ1c2VGb3JtIiwidGhlbWUiLCJjb2xvcnMi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QUE7QUFBQSxJQURGLENBRFEsR0FVUixNQUFDLGdGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUExQkosQ0FuQkosQ0FERjtBQW9ERCxDQTVITTs7R0FBTXRDLFc7VUFRSVEscUQsRUF3QzJDMEIsMEQsRUFzQjNDa0IseUNBQUksQ0FBQ0MsTzs7O0tBdEVUckQsVyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40YmYwMjQwMjhlMzhiZTVjNTQyNi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHVzZVN0YXRlLCB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCBxcyBmcm9tICdxcyc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgRm9ybSB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHtcclxuICBTdHlsZWRNb2RhbCxcclxuICBSZXN1bHRzV3JhcHBlcixcclxufSBmcm9tICcuLi92aWV3RGV0YWlsc01lbnUvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCBTZWFyY2hSZXN1bHRzIGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvc2VhcmNoL1NlYXJjaFJlc3VsdHMnO1xyXG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi9ob29rcy91c2VTZWFyY2gnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBTdHlsZWRCdXR0b24gfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi9zdHlsZXMvdGhlbWUnO1xyXG5pbXBvcnQgeyBTZWxlY3RlZERhdGEgfSBmcm9tICcuL3NlbGVjdGVkRGF0YSc7XHJcbmltcG9ydCBOYXYgZnJvbSAnLi4vTmF2JztcclxuaW1wb3J0IHsgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJztcclxuaW1wb3J0IHsgcm9vdF91cmwgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtzdG9yZX0gZnJvbSAnLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0J1xyXG5pbnRlcmZhY2UgRnJlZVNlYWNyaE1vZGFsUHJvcHMge1xyXG4gIHNldE1vZGFsU3RhdGUoc3RhdGU6IGJvb2xlYW4pOiB2b2lkO1xyXG4gIG1vZGFsU3RhdGU6IGJvb2xlYW47XHJcbiAgc2VhcmNoX3J1bl9udW1iZXI6IHVuZGVmaW5lZCB8IHN0cmluZztcclxuICBzZWFyY2hfZGF0YXNldF9uYW1lOiBzdHJpbmcgfCB1bmRlZmluZWQ7XHJcbiAgc2V0U2VhcmNoRGF0YXNldE5hbWUoZGF0YXNldF9uYW1lOiBhbnkpOiB2b2lkO1xyXG4gIHNldFNlYXJjaFJ1bk51bWJlcihydW5fbnVtYmVyOiBzdHJpbmcpOiB2b2lkO1xyXG59XHJcblxyXG5jb25zdCBvcGVuX2FfbmV3X3RhYiA9IChxdWVyeTogc3RyaW5nKSA9PiB7XHJcbiAgd2luZG93Lm9wZW4ocXVlcnksICdfYmxhbmsnKTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBTZWFyY2hNb2RhbCA9ICh7XHJcbiAgc2V0TW9kYWxTdGF0ZSxcclxuICBtb2RhbFN0YXRlLFxyXG4gIHNlYXJjaF9ydW5fbnVtYmVyLFxyXG4gIHNlYXJjaF9kYXRhc2V0X25hbWUsXHJcbiAgc2V0U2VhcmNoRGF0YXNldE5hbWUsXHJcbiAgc2V0U2VhcmNoUnVuTnVtYmVyLFxyXG59OiBGcmVlU2VhY3JoTW9kYWxQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG4gIGNvbnN0IGRhdGFzZXQgPSBxdWVyeS5kYXRhc2V0X25hbWUgPyBxdWVyeS5kYXRhc2V0X25hbWUgOiAnJztcclxuXHJcbiAgY29uc3Qge1xyXG4gICAgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXHJcbiAgICB1cGRhdGUsXHJcbiAgICBzZXRfdXBkYXRlLFxyXG4gIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcclxuXHJcbiAgY29uc29sZS5sb2codXBkYXRlKVxyXG4gIGNvbnN0IFtkYXRhc2V0TmFtZSwgc2V0RGF0YXNldE5hbWVdID0gdXNlU3RhdGUoZGF0YXNldCk7XHJcbiAgY29uc3QgW29wZW5SdW5Jbk5ld1RhYiwgdG9nZ2xlUnVuSW5OZXdUYWJdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgY29uc3QgW3J1bk51bWJlciwgc2V0UnVuTnVtYmVyXSA9IHVzZVN0YXRlPHN0cmluZz4ocnVuKTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0KTtcclxuICAgIHNldFJ1bk51bWJlcihydW4pO1xyXG4gIH0sIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXJdKTtcclxuXHJcbiAgY29uc3Qgb25DbG9zaW5nID0gKCkgPT4ge1xyXG4gICAgc2V0TW9kYWxTdGF0ZShmYWxzZSk7XHJcbiAgfTtcclxuXHJcbiAgY29uc3Qgc2VhcmNoSGFuZGxlciA9IChydW5fbnVtYmVyOiBzdHJpbmcsIGRhdGFzZXRfbmFtZTogc3RyaW5nKSA9PiB7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0X25hbWUpO1xyXG4gICAgc2V0UnVuTnVtYmVyKHJ1bl9udW1iZXIpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IG5hdmlnYXRpb25IYW5kbGVyID0gKFxyXG4gICAgc2VhcmNoX2J5X3J1bl9udW1iZXI6IHN0cmluZyxcclxuICAgIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZ1xyXG4gICkgPT4ge1xyXG4gICAgc2V0U2VhcmNoUnVuTnVtYmVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyKTtcclxuICAgIHNldFNlYXJjaERhdGFzZXROYW1lKHNlYXJjaF9ieV9kYXRhc2V0X25hbWUpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBzZWFyY2hpbmcsIGlzTG9hZGluZywgZXJyb3JzIH0gPSB1c2VTZWFyY2goXHJcbiAgICBzZWFyY2hfcnVuX251bWJlcixcclxuICAgIHNlYXJjaF9kYXRhc2V0X25hbWVcclxuICApO1xyXG5cclxuICBjb25zdCBvbk9rID0gYXN5bmMgKCkgPT4ge1xyXG4gICAgaWYgKG9wZW5SdW5Jbk5ld1RhYikge1xyXG4gICAgICBjb25zdCBwYXJhbXMgPSBmb3JtLmdldEZpZWxkc1ZhbHVlKCk7XHJcbiAgICAgIGNvbnN0IG5ld190YWJfcXVlcnlfcGFyYW1zID0gcXMuc3RyaW5naWZ5KFxyXG4gICAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhwYXJhbXMsIHF1ZXJ5KVxyXG4gICAgICApO1xyXG4gICAgICAvL3Jvb3QgdXJsIGlzIGVuZHMgd2l0aCBmaXJzdCAnPycuIEkgY2FuJ3QgdXNlIGp1c3Qgcm9vdCB1cmwgZnJvbSBjb25maWcuY29uZmlnLCBiZWNhdXNlXHJcbiAgICAgIC8vaW4gZGV2IGVudiBpdCB1c2UgbG9jYWxob3N0OjgwODEvZHFtL2RldiAodGhpcyBpcyBvbGQgYmFja2VuZCB1cmwgZnJvbSB3aGVyZSBJJ20gZ2V0dGluZyBkYXRhKSxcclxuICAgICAgLy9idXQgSSBuZWVkIGxvY2FsaG9zdDozMDAwXHJcbiAgICAgIGNvbnN0IGN1cnJlbnRfcm9vdCA9IHdpbmRvdy5sb2NhdGlvbi5ocmVmLnNwbGl0KCcvPycpWzBdO1xyXG4gICAgICBvcGVuX2FfbmV3X3RhYihgJHtjdXJyZW50X3Jvb3R9Lz8ke25ld190YWJfcXVlcnlfcGFyYW1zfWApO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgYXdhaXQgZm9ybS5zdWJtaXQoKTtcclxuICAgIH1cclxuICAgIG9uQ2xvc2luZygpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZE1vZGFsXHJcbiAgICAgIHRpdGxlPVwiU2VhcmNoIGRhdGFcIlxyXG4gICAgICB2aXNpYmxlPXttb2RhbFN0YXRlfVxyXG4gICAgICBvbkNhbmNlbD17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgIGZvb3Rlcj17W1xyXG4gICAgICAgIDxTdHlsZWRCdXR0b25cclxuICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XHJcbiAgICAgICAgICBiYWNrZ3JvdW5kPVwid2hpdGVcIlxyXG4gICAgICAgICAga2V5PVwiQ2xvc2VcIlxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgQ2xvc2VcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgICAgPFN0eWxlZEJ1dHRvbiBrZXk9XCJPS1wiIG9uQ2xpY2s9e29uT2t9PlxyXG4gICAgICAgICAgT0tcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgIF19XHJcbiAgICA+XHJcbiAgICAgIHttb2RhbFN0YXRlICYmIChcclxuICAgICAgICA8PlxyXG4gICAgICAgICAgPE5hdlxyXG4gICAgICAgICAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPXtzZWFyY2hfcnVuX251bWJlcn1cclxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtzZWFyY2hfZGF0YXNldF9uYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0RGF0YXNldE5hbWU9e2RhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0UnVuTnVtYmVyPXtydW5OdW1iZXJ9XHJcbiAgICAgICAgICAgIGhhbmRsZXI9e25hdmlnYXRpb25IYW5kbGVyfVxyXG4gICAgICAgICAgICB0eXBlPVwidG9wXCJcclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8U2VsZWN0ZWREYXRhXHJcbiAgICAgICAgICAgIGZvcm09e2Zvcm19XHJcbiAgICAgICAgICAgIGRhdGFzZXRfbmFtZT17ZGF0YXNldE5hbWV9XHJcbiAgICAgICAgICAgIHJ1bl9udW1iZXI9e3J1bk51bWJlcn1cclxuICAgICAgICAgICAgdG9nZ2xlUnVuSW5OZXdUYWI9e3RvZ2dsZVJ1bkluTmV3VGFifVxyXG4gICAgICAgICAgICBvcGVuUnVuSW5OZXdUYWI9e29wZW5SdW5Jbk5ld1RhYn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICB7c2VhcmNoaW5nID8gKFxyXG4gICAgICAgICAgICA8UmVzdWx0c1dyYXBwZXI+XHJcbiAgICAgICAgICAgICAgPFNlYXJjaFJlc3VsdHNcclxuICAgICAgICAgICAgICAgIGhhbmRsZXI9e3NlYXJjaEhhbmRsZXJ9XHJcbiAgICAgICAgICAgICAgICBpc0xvYWRpbmc9e2lzTG9hZGluZ31cclxuICAgICAgICAgICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxyXG4gICAgICAgICAgICAgICAgZXJyb3JzPXtlcnJvcnN9XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9SZXN1bHRzV3JhcHBlcj5cclxuICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgIDxSZXN1bHRzV3JhcHBlciAvPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICA8Lz5cclxuICAgICAgKX1cclxuICAgIDwvU3R5bGVkTW9kYWw+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==